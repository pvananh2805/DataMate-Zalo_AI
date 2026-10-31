from decimal import Decimal

import pytest

from datamate.query_executor import QueryExecutionError, execute_query_plan
from datamate.query_plan import (
    Aggregation,
    Calculation,
    ColumnRef,
    Filter,
    Join,
    QueryPlan,
    Sort,
    TimeRange,
)


def test_streamed_aggregation_uses_decimal_and_null_semantics() -> None:
    table = "sales"
    product = ColumnRef(table, "product")
    amount = ColumnRef(table, "amount")
    order_date = ColumnRef(table, "date")
    plan = QueryPlan(
        base_table=table,
        selected_columns=(product,),
        filters=(Filter(amount, "gt", Decimal(0)),),
        group_by=(product,),
        aggregations=(
            Aggregation("sum", amount, "revenue"),
            Aggregation("avg", amount, "average"),
            Aggregation("count", amount, "count"),
        ),
        sort=(Sort("revenue", "desc"),),
        time_range=TimeRange(order_date, "2026-01-01", "2026-01-31"),
        limit=2,
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("product", "amount", "date")},
        sources={
            table: [
                ("Tea", Decimal("10.10"), "2026-01-01"),
                ("Tea", None, "2026-01-02"),
                ("Coffee", Decimal("100"), "2026-01-03"),
                ("Tea", Decimal("2.30"), "2026-02-01"),
            ]
        },
    )

    assert result.rows == (
        {
            "sales.product": "Coffee",
            "revenue": Decimal("100"),
            "average": Decimal("100"),
            "count": 1,
        },
        {
            "sales.product": "Tea",
            "revenue": Decimal("10.10"),
            "average": Decimal("10.10"),
            "count": 1,
        },
    )
    assert result.trace.source_rows == {table: 4}
    assert result.trace.joined_rows == 4
    assert result.trace.filtered_rows == 2
    assert result.trace.group_count == 2


def test_left_join_keeps_unmatched_rows_and_aggregates_exactly() -> None:
    orders = "orders"
    customers = "customers"
    customer_name = ColumnRef(customers, "name")
    amount = ColumnRef(orders, "amount")
    plan = QueryPlan(
        base_table=orders,
        joins=(
            Join(
                ColumnRef(orders, "customer_id"),
                ColumnRef(customers, "id"),
                kind="left",
            ),
        ),
        group_by=(customer_name,),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        sort=(Sort("revenue", "desc"),),
    )

    result = execute_query_plan(
        plan,
        catalog={
            orders: ("customer_id", "amount"),
            customers: ("id", "name"),
        },
        sources={
            orders: [(1, Decimal("10")), (2, Decimal("5")), (3, Decimal("7"))],
            customers: [(1, "An"), (2, "Bình")],
        },
    )

    assert [row["revenue"] for row in result.rows] == [
        Decimal("10"),
        Decimal("7"),
        Decimal("5"),
    ]
    assert result.rows[1][f"{customers}.name"] is None
    assert result.trace.source_rows == {customers: 2, orders: 3}
    assert result.trace.joined_rows == 3


def test_count_star_and_null_filter_work_for_empty_numeric_columns() -> None:
    table = "items"
    value = ColumnRef(table, "value")
    plan = QueryPlan(
        base_table=table,
        group_by=(value,),
        aggregations=(Aggregation("count", None, "rows"),),
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("value",)},
        sources={table: [(None,), (None,), ("x",)]},
    )

    assert result.rows == (
        {f"{table}.value": None, "rows": 2},
        {f"{table}.value": "x", "rows": 1},
    )


def test_executor_rejects_non_numeric_values_and_configured_limit_overflow() -> None:
    table = "sales"
    amount = ColumnRef(table, "amount")
    aggregate = QueryPlan(
        base_table=table,
        aggregations=(Aggregation("sum", amount, "revenue"),),
    )
    kwargs = {"catalog": {table: ("amount",)}}

    with pytest.raises(QueryExecutionError, match="requires numeric values"):
        execute_query_plan(aggregate, sources={table: [("12,50",)]}, **kwargs)

    grouped = QueryPlan(base_table=table, group_by=(amount,))
    with pytest.raises(QueryExecutionError, match="Group count exceeds"):
        execute_query_plan(
            grouped,
            sources={table: [(1,), (2,)]},
            max_groups=1,
            **kwargs,
        )


def test_large_decimal_sum_does_not_round_to_default_context_precision() -> None:
    table = "ledger"
    amount = ColumnRef(table, "amount")
    plan = QueryPlan(
        base_table=table,
        aggregations=(Aggregation("sum", amount, "total"),),
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("amount",)},
        sources={
            table: [
                (Decimal("9999999999999999999999999999"),),
                (Decimal("1"),),
            ]
        },
    )

    assert result.rows[0]["total"] == Decimal("10000000000000000000000000000")


def test_decimal_addition_preserves_small_terms_at_large_exponents() -> None:
    table = "ledger"
    amount = ColumnRef(table, "amount")
    plan = QueryPlan(
        base_table=table,
        aggregations=(Aggregation("sum", amount, "total"),),
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("amount",)},
        sources={table: [(Decimal("1E+100"),), (Decimal("1"),)]},
    )

    assert result.rows[0]["total"] == Decimal("1" + "0" * 99 + "1")


def test_ratio_and_percent_change_are_deterministic_and_guard_zero_division() -> None:
    table = "periods"
    current = ColumnRef(table, "current")
    previous = ColumnRef(table, "previous")
    plan = QueryPlan(
        base_table=table,
        aggregations=(
            Aggregation("sum", current, "current_total"),
            Aggregation("sum", previous, "previous_total"),
        ),
        calculations=(
            Calculation(
                "percent_change",
                "current_total",
                "previous_total",
                "growth_percent",
            ),
            Calculation(
                "ratio",
                "current_total",
                "previous_total",
                "ratio",
            ),
        ),
    )
    arguments = {
        "catalog": {table: ("current", "previous")},
        "sources": {table: [(Decimal("120"), Decimal("100"))]},
    }

    result = execute_query_plan(plan, **arguments)

    assert result.rows[0]["growth_percent"] == Decimal("20")
    assert result.rows[0]["ratio"] == Decimal("1.2")
    with pytest.raises(QueryExecutionError, match="divides by zero"):
        execute_query_plan(
            plan,
            catalog=arguments["catalog"],
            sources={table: [(Decimal("120"), Decimal("0"))]},
        )
