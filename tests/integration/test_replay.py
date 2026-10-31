import json
import sqlite3
from pathlib import Path

import pytest

from datamate.data_loader import BusinessDataLoader
from datamate.normalization import ColumnCoercion
from datamate.pipeline import (
    execute_and_log_business_plan,
    execute_question_and_log,
    table_identifier,
)
from datamate.query_plan import Aggregation, ColumnRef, GroupFilter, QueryPlan
from datamate.replay import ReplayError, replay_run_record
from datamate.question_understanding import JsonPlanAdapter


def _business_directory(root: Path) -> Path:
    business = root / "biz-001"
    business.mkdir(parents=True)
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute("CREATE TABLE sales (sale_id INTEGER, amount NUMERIC)")
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?)",
            [(1, 12), (2, 8)],
        )
    return business


def _record_successful_run(root: Path, log_path: Path) -> tuple[Path, str]:
    business = _business_directory(root)
    table = next(
        table
        for table in BusinessDataLoader(business).discover()
        if table.name == "sales"
    )
    table_id = table_identifier(table)
    amount = ColumnRef(table_id, "amount")
    execute_and_log_business_plan(
        business,
        QueryPlan(
            base_table=table_id,
            aggregations=(Aggregation("sum", amount, "total"),),
            output_type="number",
        ),
        question_id="q-1",
        business_id="biz-001",
        question="Tổng tiền là bao nhiêu?",
        model="manual-plan",
        log_path=log_path,
        input_root=root,
        column_coercions={amount: ColumnCoercion("number")},
    )
    return business, table_id


def test_replay_reproduces_answer_trace_and_logged_coercions(tmp_path: Path) -> None:
    business_root = tmp_path / "businesses"
    business_root.mkdir()
    log_path = tmp_path / "logs" / "runs.jsonl"
    _record_successful_run(business_root, log_path)

    replay = replay_run_record(log_path, "q-1", business_root)

    assert replay["answer_matches"] is True
    assert replay["execution_trace_matches"] is True
    assert replay["replay_matches"] is True


def test_replay_reproduces_group_threshold_plan(tmp_path: Path) -> None:
    business_root = tmp_path / "businesses"
    business_root.mkdir()
    business = business_root / "biz-001"
    business.mkdir()
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute("CREATE TABLE sales (product TEXT, order_id TEXT)")
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?)",
            [("A", "O-1"), ("A", "O-2"), ("B", "O-1")],
        )
    table = next(
        table
        for table in BusinessDataLoader(business).discover()
        if table.name == "sales"
    )
    table_id = table_identifier(table)
    product = ColumnRef(table_id, "product")
    order_id = ColumnRef(table_id, "order_id")
    plan = QueryPlan(
        base_table=table_id,
        group_by=(product,),
        group_filters=(GroupFilter("orders", "gte", 2),),
        aggregations=(Aggregation("count_distinct", order_id, "orders"),),
        count_groups=True,
        output_type="number",
    )
    log_path = tmp_path / "logs" / "group-runs.jsonl"

    execute_and_log_business_plan(
        business,
        plan,
        question_id="q-group-count",
        business_id="biz-001",
        question="Bao nhiêu sản phẩm có ít nhất hai đơn?",
        model="manual-plan",
        log_path=log_path,
        input_root=business_root,
    )

    replay = replay_run_record(log_path, "q-group-count", business_root)

    assert replay["answer_matches"] is True
    assert replay["execution_trace_matches"] is True
    assert replay["replay_matches"] is True


def test_replay_reports_changed_source_data_as_mismatch(tmp_path: Path) -> None:
    business_root = tmp_path / "businesses"
    business_root.mkdir()
    log_path = tmp_path / "logs" / "runs.jsonl"
    business, _ = _record_successful_run(business_root, log_path)
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute("UPDATE sales SET amount = 99 WHERE sale_id = 1")

    replay = replay_run_record(log_path, "q-1", business_root)

    assert replay["answer_matches"] is False
    assert replay["execution_trace_matches"] is True
    assert replay["replay_matches"] is False


def test_replay_rejects_logs_without_execution_limits(tmp_path: Path) -> None:
    business_root = tmp_path / "businesses"
    business_root.mkdir()
    log_path = tmp_path / "logs" / "runs.jsonl"
    _record_successful_run(business_root, log_path)
    record = json.loads(log_path.read_text(encoding="utf-8"))
    record["input"]["metadata"].pop("execution_limits")
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ReplayError, match="lacks execution limits"):
        replay_run_record(log_path, "q-1", business_root)


@pytest.mark.parametrize(
    ("output_type", "status"),
    [("unanswerable", "ambiguous"), ("abstain", "abstained")],
)
def test_replay_validates_abstention_without_execution(
    tmp_path: Path,
    output_type: str,
    status: str,
) -> None:
    business_root = tmp_path / "businesses"
    business_root.mkdir()
    business = _business_directory(business_root)
    base_table = table_identifier(
        next(
            table
            for table in BusinessDataLoader(business).discover()
            if table.name == "sales"
        )
    )
    log_path = tmp_path / "logs" / "runs.jsonl"
    plan = {
        "base_table": base_table,
        "output_type": output_type,
        "metadata": {
            "status": status,
            "reason": "Câu hỏi chưa đủ rõ để trả lời.",
        },
    }
    execute_question_and_log(
        business_root / "biz-001",
        "Câu hỏi mơ hồ",
        JsonPlanAdapter(
            lambda _: json.dumps(plan, ensure_ascii=False),
            model_name="test-planner",
        ),
        question_id="q-abstain",
        business_id="biz-001",
        as_of_date=None,
        log_path=log_path,
        input_root=business_root,
    )

    replay = replay_run_record(log_path, "q-abstain", business_root)

    assert replay["answer_matches"] is True
    assert replay["execution_trace_matches"] is True
    assert replay["replay_matches"] is True
