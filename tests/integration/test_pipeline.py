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
    Calculation,
    ColumnRef,
    Filter,
    PlanValidationError,
    QueryPlan,
    Sort,
    TimeBucket,
    TimeRange,
)
from datamate.run_logging import RunLogError, read_run_records
from datamate.submission import SubmissionWriter, serialize_submission_record


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


def test_pipeline_compares_period_scoped_aggregations_and_records_evidence(
    tmp_path: Path,
) -> None:
    business = tmp_path / "period-business"
    business.mkdir()
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE sales (product TEXT, amount NUMERIC, sold_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?, ?)",
            [
                ("A", 60, "2026-01-10"),
                ("A", 40, "2026-01-31"),
                ("A", 120, "2026-02-28T23:59:00"),
                ("B", 50, "2026-01-15"),
                ("B", 25, "2026-02-10"),
                ("C", 500, "2026-03-10"),
            ],
        )
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    product = ColumnRef(table_id, "product")
    amount = ColumnRef(table_id, "amount")
    sold_at = ColumnRef(table_id, "sold_at")
    plan = QueryPlan(
        base_table=table_id,
        group_by=(product,),
        aggregations=(
            Aggregation(
                "sum",
                amount,
                "previous_revenue",
                time_range=TimeRange(sold_at, "2026-01-01", "2026-01-31"),
            ),
            Aggregation(
                "sum",
                amount,
                "current_revenue",
                time_range=TimeRange(sold_at, "2026-02-01", "2026-02-28"),
            ),
        ),
        calculations=(
            Calculation(
                "percent_change",
                "current_revenue",
                "previous_revenue",
                "growth_percent",
            ),
        ),
        sort=(Sort(product, "asc"),),
        output_type="table",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.answer_type == "table"
    assert result.answer.value == [
        {
            f"{table_id}.product": "A",
            "previous_revenue": 100,
            "current_revenue": 120,
            "growth_percent": 20,
        },
        {
            f"{table_id}.product": "B",
            "previous_revenue": 50,
            "current_revenue": 25,
            "growth_percent": -50,
        },
    ]
    assert result.answer.evidence[0].columns == ("amount", "product", "sold_at")
    assert result.answer.evidence[0].filter_summary == "0 filters; time_range=yes"


def test_pipeline_groups_by_month_and_serializes_table_with_bucket_evidence(
    tmp_path: Path,
) -> None:
    business = tmp_path / "monthly-business"
    business.mkdir()
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE sales (amount NUMERIC, sold_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?)",
            [
                (10, "2026-01-01"),
                (15, "2026-01-31T23:59:00"),
                (25, "2026-02-01"),
                (100, "2026-03-01"),
            ],
        )
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    amount = ColumnRef(table_id, "amount")
    sold_at = ColumnRef(table_id, "sold_at")
    plan = QueryPlan(
        base_table=table_id,
        time_range=TimeRange(sold_at, "2026-01-01", "2026-02-28"),
        time_buckets=(TimeBucket(sold_at, "month", "month"),),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        sort=(Sort("month", "asc"),),
        output_type="table",
    )

    result = execute_business_plan(business, plan)
    submission = json.loads(
        serialize_submission_record(
            "q-monthly",
            result.answer,
            plan=plan,
        )
    )

    assert submission == {
        "id": "q-monthly",
        "type": "table",
        "answer": {
            "columns": ["month", "revenue"],
            "rows": [["2026-01", 25], ["2026-02", 25]],
        },
    }
    assert result.answer.evidence[0].columns == ("amount", "sold_at")
    assert result.answer.evidence[0].filter_summary == (
        "0 filters; time_range=yes; time_bucket=yes"
    )

    list_plan = QueryPlan(
        base_table=table_id,
        time_range=TimeRange(sold_at, "2026-01-01", "2026-02-28"),
        time_buckets=(TimeBucket(sold_at, "month", "month"),),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        sort=(Sort("month", "asc"),),
        output_type="list",
    )
    list_result = execute_business_plan(business, list_plan)
    assert list_result.answer.value == [25, 25]
    list_submission = json.loads(
        serialize_submission_record(
            "q-monthly-series",
            list_result.answer,
            plan=list_plan,
        )
    )
    assert list_submission == {
        "id": "q-monthly-series",
        "type": "list",
        "answer": [25, 25],
    }


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
    prompts: list[str] = []
    planner = JsonPlanAdapter(
        lambda prompt: prompts.append(prompt) or json.dumps(generated_plan),
        model_name="local-stub",
    )
    log_path = tmp_path / "logs" / "question.jsonl"
    submission_path = tmp_path / "result" / "submission.jsonl"

    result = execute_question_and_log(
        business,
        "Tổng doanh thu là bao nhiêu?",
        planner,
        question_id="q-003",
        business_id="biz-test",
        as_of_date="2026-01-31",
        log_path=log_path,
        input_root=input_root,
        rulebook="# Quy tắc doanh thu",
        business_readme="amount tính bằng VND",
        calendar_columns=("solar_date",),
        calendar_rows=({"solar_date": "2026-01-31"},),
        input_metadata={
            "business_context_sources": {
                "rulebook": {"status": "loaded", "sha256": "abc123"}
            }
        },
    )
    writer = SubmissionWriter(submission_path)
    writer.append("q-003", result.answer, plan=result.plan)
    writer.commit()

    (record,) = read_run_records(log_path)
    assert result.answer.value == 20
    assert record["model"] == "local-stub"
    assert record["input"]["metadata"]["as_of_date"] == "2026-01-31"
    assert record["input"]["metadata"]["business_context_sources"]["rulebook"][
        "sha256"
    ] == "abc123"
    assert record["query_plan"]["base_table"] == table_id
    assert prompts and "# Quy tắc doanh thu" in prompts[0]
    assert "amount tính bằng VND" in prompts[0]
    assert "2026-01-31" in prompts[0]
    assert json.loads(submission_path.read_text(encoding="utf-8")) == {
        "id": "q-003",
        "type": "number",
        "answer": 20,
    }


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


@pytest.mark.parametrize(
    ("output_type", "status"),
    [("unanswerable", "ambiguous"), ("abstain", "abstained")],
)
def test_question_pipeline_records_explicit_abstention_without_execution(
    tmp_path: Path,
    output_type: str,
    status: str,
) -> None:
    input_root = tmp_path / "input"
    business = _business_database(input_root)
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    abstention = {
        "base_table": table_id,
        "output_type": output_type,
        "metadata": {
            "status": status,
            "reason": "Chưa rõ người hỏi muốn tổng tiền hay số hóa đơn.",
        },
    }
    log_path = tmp_path / "logs" / "abstention.jsonl"

    result = execute_question_and_log(
        business,
        "Tháng này thế nào?",
        JsonPlanAdapter(
            lambda _: json.dumps(abstention),
            model_name="local-stub",
        ),
        question_id="q-ambiguous",
        business_id="biz-test",
        as_of_date="2026-01-31",
        log_path=log_path,
        input_root=input_root,
    )

    (record,) = read_run_records(log_path)
    assert result.answer.status == status
    assert result.answer.answer_type == output_type
    assert result.execution is None
    assert record["execution_trace"] is None
    assert record["answer"]["value"] is None
    assert record["answer"]["explanation"] == abstention["metadata"]["reason"]


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
