import json
import sqlite3
from pathlib import Path

import pytest

from datamate.pipeline import (
    execute_and_log_business_plan,
    execute_business_plan,
    execute_question_and_log,
    table_identifier,
)
from datamate.data_loader import BusinessDataLoader
from datamate.normalization import ColumnCoercion
from datamate.question_understanding import (
    JsonPlanAdapter,
    QuestionUnderstandingError,
)
from datamate.query_executor import QueryExecutionError
from datamate.query_plan import (
    Aggregation,
    ColumnRef,
    Filter,
    PlanValidationError,
    QueryPlan,
)
from datamate.run_logging import RunLogError, read_run_records


def _business_database(tmp_path: Path) -> Path:
    business = tmp_path / "business"
    business.mkdir(parents=True)
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE sales (sale_id INTEGER PRIMARY KEY, amount NUMERIC)"
        )
        connection.executemany("INSERT INTO sales (amount) VALUES (?)", [(12,), (8,)])
    return business


def test_business_plan_runs_from_read_only_loader_through_answer_validation(
    tmp_path: Path,
) -> None:
    business = _business_database(tmp_path)
    loader = BusinessDataLoader(business)
    table = next(table for table in loader.discover() if table.name == "sales")
    table_id = table_identifier(table)
    amount = ColumnRef(table_id, "amount")
    plan = QueryPlan(
        base_table=table_id,
        aggregations=(Aggregation("sum", amount, "total"),),
        output_type="number",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.value == 20
    assert result.answer.answer_type == "number"
    assert result.answer.status == "answered"
    assert result.answer.evidence[0].source_table == table_id
    assert result.execution.trace.source_rows == {table_id: 2}


def test_pipeline_distinguishes_no_matching_data_from_internal_failure(
    tmp_path: Path,
) -> None:
    business = _business_database(tmp_path)
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    amount = ColumnRef(table_id, "amount")
    plan = QueryPlan(
        base_table=table_id,
        filters=(Filter(amount, "gt", 100),),
        aggregations=(Aggregation("sum", amount, "total"),),
        output_type="number",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.answer_type == "unanswerable"
    assert result.answer.status == "no_data"
    assert result.answer.value is None
    assert result.answer.evidence[0].row_count == 2


def test_pipeline_preserves_exact_zero_count_for_no_matching_rows(
    tmp_path: Path,
) -> None:
    business = _business_database(tmp_path)
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    amount = ColumnRef(table_id, "amount")
    plan = QueryPlan(
        base_table=table_id,
        filters=(Filter(amount, "gt", 100),),
        aggregations=(Aggregation("count", None, "rows"),),
        output_type="number",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.answer_type == "number"
    assert result.answer.value == 0
    assert result.answer.status == "answered"
    assert result.execution.trace.filtered_rows == 0


def test_csv_plan_uses_explicit_locale_normalization_without_guessing(
    tmp_path: Path,
) -> None:
    business = tmp_path / "csv-business"
    business.mkdir()
    (business / "sales.csv").write_text(
        "amount;note\n1.234,50;first\n2,50;second\n",
        encoding="utf-8",
    )
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    amount = ColumnRef(table_id, "amount")
    plan = QueryPlan(
        base_table=table_id,
        aggregations=(Aggregation("sum", amount, "total"),),
        output_type="number",
    )

    with pytest.raises(QueryExecutionError, match="requires numeric values"):
        execute_business_plan(business, plan)

    result = execute_business_plan(
        business,
        plan,
        column_coercions={
            amount: ColumnCoercion(
                "number",
                decimal_separator=",",
                thousands_separator=".",
            )
        },
    )

    assert result.answer.value == 1237
    assert result.answer.status == "answered"


def test_pipeline_logs_success_and_rejects_logs_inside_input_root(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    business = _business_database(input_root)
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    amount = ColumnRef(table_id, "amount")
    plan = QueryPlan(
        base_table=table_id,
        aggregations=(Aggregation("sum", amount, "total"),),
        output_type="number",
    )
    log_path = tmp_path / "logs" / "runs.jsonl"

    result = execute_and_log_business_plan(
        business,
        plan,
        question_id="q-001",
        business_id="biz-001",
        question="Tổng tiền là bao nhiêu?",
        model="manual-query-plan",
        log_path=log_path,
        input_root=input_root,
        input_metadata={"as_of_date": "2026-01-31"},
        column_coercions={amount: ColumnCoercion("number")},
    )

    (record,) = read_run_records(log_path)
    assert result.answer.value == 20
    assert record["question_id"] == "q-001"
    assert record["answer"]["value"] == "20"
    assert record["execution_trace"]["result_count"] == 1
    assert record["input"]["metadata"]["column_coercions"][
        f"{table_id}.amount"
    ] == {
        "kind": "number",
        "decimal_separator": ".",
        "thousands_separator": None,
        "date_format": None,
    }
    with pytest.raises(RunLogError, match="must not be written under input root"):
        execute_and_log_business_plan(
            business,
            plan,
            question_id="q-002",
            business_id="biz-001",
            question="Tổng tiền?",
            model="manual-query-plan",
            log_path=input_root / "runs.jsonl",
            input_root=input_root,
        )


def test_question_to_plan_to_answer_is_integrated_and_logged(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    business = _business_database(input_root)
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    generated_plan = {
        "base_table": table_id,
        "aggregations": [
            {
                "function": "sum",
                "column": {"table": table_id, "column": "amount"},
                "alias": "total",
            }
        ],
        "output_type": "number",
    }
    planner = JsonPlanAdapter(
        lambda prompt: json.dumps(generated_plan),
        model_name="local-stub",
    )
    log_path = tmp_path / "logs" / "question.jsonl"

    result = execute_question_and_log(
        business,
        "Tổng doanh thu là bao nhiêu?",
        planner,
        question_id="q-003",
        business_id="biz-test",
        as_of_date="2026-01-31",
        log_path=log_path,
        input_root=input_root,
    )

    (record,) = read_run_records(log_path)
    assert result.answer.value == 20
    assert record["model"] == "local-stub"
    assert record["input"]["metadata"]["as_of_date"] == "2026-01-31"
    assert record["query_plan"]["base_table"] == table_id


def test_question_planning_failure_is_recorded_without_a_fake_plan(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    business = _business_database(input_root)
    log_path = tmp_path / "logs" / "question-failure.jsonl"

    with pytest.raises(QuestionUnderstandingError, match="Generated query plan is invalid"):
        execute_question_and_log(
            business,
            "Câu hỏi không thể lập kế hoạch",
            JsonPlanAdapter(lambda _: "not JSON", model_name="local-stub"),
            question_id="q-bad-plan",
            business_id="biz-test",
            as_of_date=None,
            log_path=log_path,
            input_root=input_root,
        )

    (record,) = read_run_records(log_path)
    assert record["query_plan"] is None
    assert record["execution_trace"] is None
    assert "QuestionUnderstandingError" in record["error"]


def test_pipeline_records_plan_validation_failures(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    business = _business_database(input_root)
    log_path = tmp_path / "logs" / "failures.jsonl"
    plan = QueryPlan(base_table="missing.csv::table", output_type="number")

    with pytest.raises(PlanValidationError):
        execute_and_log_business_plan(
            business,
            plan,
            question_id="q-bad",
            business_id="biz-001",
            question="Câu hỏi không thể lập plan",
            model="manual-query-plan",
            log_path=log_path,
            input_root=input_root,
        )

    (record,) = read_run_records(log_path)
    assert record["answer"] is None
    assert "PlanValidationError" in record["error"]
