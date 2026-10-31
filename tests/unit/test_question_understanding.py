import json

import pytest

from datamate.plan_serialization import (
    QueryPlanFormatError,
    query_plan_from_json,
)
from datamate.question_understanding import (
    JsonPlanAdapter,
    QuestionContext,
    QuestionUnderstandingError,
    build_planning_prompt,
)


TABLE = "sales.csv::table"
PLAN = {
    "base_table": TABLE,
    "aggregations": [
        {
            "function": "sum",
            "column": {"table": TABLE, "column": "amount"},
            "alias": "total",
        }
    ],
    "output_type": "number",
}


def test_query_plan_json_decodes_supported_nested_fields() -> None:
    payload = {
        **PLAN,
        "filters": [
            {
                "column": {"table": TABLE, "column": "amount"},
                "operator": "between",
                "value": [10, 25],
            }
        ],
        "time_range": {
            "column": {"table": TABLE, "column": "sold_at"},
            "start": "2026-01-01",
            "end": "2026-01-31",
            "inclusive_end": False,
        },
        "aggregations": [
            {
                **PLAN["aggregations"][0],
                "time_range": {
                    "column": {"table": TABLE, "column": "sold_at"},
                    "start": "2025-12-01",
                    "end": "2025-12-31",
                },
                "filters": [
                    {
                        "column": {"table": TABLE, "column": "status"},
                        "operator": "eq",
                        "value": "completed",
                    }
                ],
            }
        ],
        "metadata": {"reason": "monthly total"},
    }

    plan = query_plan_from_json(json.dumps(payload))

    assert plan.filters[0].value == (10, 25)
    assert plan.time_range is not None
    assert plan.time_range.inclusive_end is False
    aggregation = plan.aggregations[0]
    assert aggregation.time_range is not None
    assert aggregation.time_range.start == "2025-12-01"
    assert aggregation.filters[0].value == "completed"
    assert plan.metadata["reason"] == "monthly total"


@pytest.mark.parametrize(
    "raw",
    [
        '{"base_table":"sales.csv::table","unexpected":true}',
        '{"base_table":"sales.csv::table","limit":true}',
        '{"base_table":"sales.csv::table","base_table":"other::table"}',
        '{"base_table":"sales.csv::table","filters":[{"column":{"table":"sales.csv::table","column":"amount"},"operator":2,"value":10}]}',
        '{"base_table":"sales.csv::table","value":NaN}',
    ],
)
def test_query_plan_json_rejects_malformed_or_unsafe_shapes(raw: str) -> None:
    with pytest.raises(QueryPlanFormatError):
        query_plan_from_json(raw)


def test_planning_prompt_carries_schema_and_reference_date() -> None:
    prompt = build_planning_prompt(
        "doanh thu tháng này",
        QuestionContext(
            schema={TABLE: ("amount", "sold_at")},
            as_of_date="2026-05-25",
            business_rules=("Hủy đơn không tính doanh thu.",),
            rulebook="# Nguyên tắc",
            business_readme="amount tính bằng VND",
            calendar_columns=("solar_date", "lunar_date"),
            calendar_rows=(
                {"solar_date": "2026-05-25", "lunar_date": "2026-04-09"},
            ),
        ),
    )

    assert "doanh thu tháng này" in prompt
    assert "2026-05-25" in prompt
    assert "sold_at" in prompt
    assert "Hủy đơn không tính doanh thu." in prompt
    assert "# Nguyên tắc" in prompt
    assert "amount tính bằng VND" in prompt
    assert "2026-04-09" in prompt
    assert "business_rules hiện chưa có executor" in prompt


def test_json_plan_adapter_validates_against_discovered_schema() -> None:
    prompts: list[str] = []
    adapter = JsonPlanAdapter(
        lambda prompt: prompts.append(prompt) or json.dumps(PLAN),
        model_name="local-test",
    )

    plan = adapter.create_plan(
        "Tổng doanh thu?",
        QuestionContext(schema={TABLE: ("amount",)}),
    )

    assert plan.base_table == TABLE
    assert prompts and "Tổng doanh thu?" in prompts[0]


def test_json_plan_adapter_rejects_hallucinated_columns_and_non_json() -> None:
    hallucinated = {
        **PLAN,
        "aggregations": [
            {
                "function": "sum",
                "column": {"table": TABLE, "column": "missing"},
                "alias": "total",
            }
        ],
    }
    context = QuestionContext(schema={TABLE: ("amount",)})

    with pytest.raises(QuestionUnderstandingError, match="unknown column"):
        JsonPlanAdapter(
            lambda _: json.dumps(hallucinated),
            model_name="local-test",
        ).create_plan("Tổng?", context)
    with pytest.raises(QuestionUnderstandingError, match="valid JSON"):
        JsonPlanAdapter(
            lambda _: "```json\n{}\n```",
            model_name="local-test",
        ).create_plan("Tổng?", context)


def test_json_plan_adapter_rejects_unexecutable_business_rule_tags() -> None:
    tagged_plan = {
        **PLAN,
        "business_rules": ["exclude cancelled orders"],
    }

    with pytest.raises(QuestionUnderstandingError, match="business_rules are not executable"):
        JsonPlanAdapter(
            lambda _: json.dumps(tagged_plan),
            model_name="local-test",
        ).create_plan("Tổng doanh thu?", QuestionContext(schema={TABLE: ("amount",)}))


def test_json_plan_adapter_accepts_only_explicit_non_answer_statuses() -> None:
    context = QuestionContext(schema={TABLE: ("amount",)})
    ambiguous_plan = {
        "base_table": TABLE,
        "output_type": "unanswerable",
        "metadata": {
            "status": "ambiguous",
            "reason": "Chưa rõ cần tính doanh thu hay số đơn.",
        },
    }

    plan = JsonPlanAdapter(
        lambda _: json.dumps(ambiguous_plan),
        model_name="local-test",
    ).create_plan("Câu hỏi mơ hồ", context)

    assert plan.output_type == "unanswerable"
    assert plan.metadata["status"] == "ambiguous"
    ambiguous_plan["metadata"]["status"] = "no_data"
    with pytest.raises(QuestionUnderstandingError, match="metadata.status"):
        JsonPlanAdapter(
            lambda _: json.dumps(ambiguous_plan),
            model_name="local-test",
        ).create_plan("Câu hỏi mơ hồ", context)
