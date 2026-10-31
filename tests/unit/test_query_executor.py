from datetime import date, datetime
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
    decreasing = execute_query_plan(
        plan,
        catalog=arguments["catalog"],
        sources={table: [(Decimal("80"), Decimal("100"))]},
    )
    assert decreasing.rows[0]["growth_percent"] == Decimal("-20")
    with pytest.raises(QueryExecutionError, match="divides by zero"):
        execute_query_plan(
            plan,
            catalog=arguments["catalog"],
            sources={table: [(Decimal("120"), Decimal("0"))]},
        )


def test_percentage_calculations_round_half_up_to_two_decimal_places() -> None:
    table = "periods"
    numerator = ColumnRef(table, "numerator")
    denominator = ColumnRef(table, "denominator")
    plan = QueryPlan(
        base_table=table,
        aggregations=(
            Aggregation("sum", numerator, "numerator"),
            Aggregation("sum", denominator, "denominator"),
        ),
        calculations=(
            Calculation("ratio_percent", "numerator", "denominator", "share"),
            Calculation(
                "percent_change",
                "numerator",
                "denominator",
                "growth_percent",
            ),
        ),
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("numerator", "denominator")},
        sources={table: [(Decimal(1), Decimal(3))]},
    )

    assert result.rows[0]["share"] == Decimal("33.33")
    assert result.rows[0]["growth_percent"] == Decimal("-66.67")

    half_cent_boundary = execute_query_plan(
        plan,
        catalog={table: ("numerator", "denominator")},
        sources={table: [(Decimal(1), Decimal(32))]},
    )
    assert half_cent_boundary.rows[0]["share"] == Decimal("3.13")
    assert half_cent_boundary.rows[0]["growth_percent"] == Decimal("-96.88")


def test_min_max_and_top_k_honor_explicit_tie_break_sort() -> None:
    table = "sales"
    product_code = ColumnRef(table, "product_code")
    quantity = ColumnRef(table, "quantity")
    plan = QueryPlan(
        base_table=table,
        group_by=(product_code,),
        aggregations=(
            Aggregation("sum", quantity, "units_sold"),
            Aggregation("min", quantity, "minimum_sale"),
            Aggregation("max", quantity, "maximum_sale"),
        ),
        sort=(Sort("units_sold", "desc"), Sort(product_code, "asc")),
        limit=2,
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("product_code", "quantity")},
        sources={
            table: [
                ("P-02", Decimal(5)),
                ("P-01", Decimal(5)),
                ("P-03", Decimal(4)),
                ("P-01", Decimal(2)),
            ]
        },
    )

    assert result.rows == (
        {
            f"{table}.product_code": "P-01",
            "units_sold": Decimal(7),
            "minimum_sale": Decimal(2),
            "maximum_sale": Decimal(5),
        },
        {
            f"{table}.product_code": "P-02",
            "units_sold": Decimal(5),
            "minimum_sale": Decimal(5),
            "maximum_sale": Decimal(5),
        },
    )


def test_scoped_aggregations_compare_two_periods_from_the_same_table() -> None:
    table = "sales"
    product = ColumnRef(table, "product")
    amount = ColumnRef(table, "amount")
    sold_at = ColumnRef(table, "sold_at")
    plan = QueryPlan(
        base_table=table,
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
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("product", "amount", "sold_at")},
        sources={
            table: [
                ("A", Decimal(60), date(2026, 1, 10)),
                ("A", Decimal(40), date(2026, 1, 31)),
                ("A", Decimal(120), datetime(2026, 2, 28, 23, 59)),
                ("B", Decimal(50), date(2026, 1, 15)),
                ("B", Decimal(25), date(2026, 2, 10)),
                ("C", Decimal(500), date(2026, 3, 10)),
            ]
        },
    )

    assert result.rows == (
        {
            f"{table}.product": "A",
            "previous_revenue": Decimal(100),
            "current_revenue": Decimal(120),
            "growth_percent": Decimal(20),
        },
        {
            f"{table}.product": "B",
            "previous_revenue": Decimal(50),
            "current_revenue": Decimal(25),
            "growth_percent": Decimal(-50),
        },
    )


@pytest.mark.parametrize(
    ("values", "inclusive_end", "expected"),
    [
        (
            [date(2026, 1, 1), date(2026, 1, 31), date(2026, 2, 1)],
            True,
            [date(2026, 1, 1), date(2026, 1, 31)],
        ),
        (
            [
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 31, 23, 59, 59),
                datetime(2026, 2, 1, 0, 0),
            ],
            True,
            [datetime(2026, 1, 1, 0, 0), datetime(2026, 1, 31, 23, 59, 59)],
        ),
        (
            [
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 31, 0, 0),
                datetime(2026, 2, 1, 0, 0),
            ],
            False,
            [datetime(2026, 1, 1, 0, 0)],
        ),
        (
            ["2026-01-01T00:00:00", "2026-01-31T23:59:59", "2026-02-01"],
            True,
            ["2026-01-01T00:00:00", "2026-01-31T23:59:59"],
        ),
    ],
)
def test_time_range_handles_date_and_datetime_cells_by_calendar_day(
    values: list[object],
    inclusive_end: bool,
    expected: list[object],
) -> None:
    table = "orders"
    order_date = ColumnRef(table, "order_date")
    plan = QueryPlan(
        base_table=table,
        selected_columns=(order_date,),
        time_range=TimeRange(
            order_date,
            "2026-01-01",
            "2026-01-31",
            inclusive_end=inclusive_end,
        ),
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("order_date",)},
        sources={table: [(value,) for value in values]},
    )

    assert [row[f"{table}.order_date"] for row in result.rows] == expected


def test_time_range_rejects_non_iso_date_values_instead_of_lexically_filtering() -> None:
    table = "orders"
    order_date = ColumnRef(table, "order_date")
    plan = QueryPlan(
        base_table=table,
        selected_columns=(order_date,),
        time_range=TimeRange(order_date, "2026-01-01", "2026-01-31"),
    )

    with pytest.raises(QueryExecutionError, match="must be an ISO-8601 date"):
        execute_query_plan(
            plan,
            catalog={table: ("order_date",)},
            sources={table: [("31/01/2026",)]},
        )
