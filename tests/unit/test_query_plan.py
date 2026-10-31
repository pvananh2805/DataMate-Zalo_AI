import pytest

from datamate.query_plan import (
    Aggregation,
    Calculation,
    ColumnRef,
    Filter,
    Join,
    PlanValidationError,
    QueryPlan,
    Sort,
    TimeRange,
    validate_query_plan,
)


CATALOG = {
    "orders.csv::orders": {"order_id", "customer_id", "amount", "order_date"},
    "customers.xlsx::Customers": {"customer_id", "name"},
}


def test_valid_plan_covers_joins_aggregation_time_and_evidence() -> None:
    orders = "orders.csv::orders"
    customers = "customers.xlsx::Customers"
    plan = QueryPlan(
        base_table=orders,
        joins=(
            Join(
                ColumnRef(orders, "customer_id"),
                ColumnRef(customers, "customer_id"),
                kind="left",
            ),
        ),
        filters=(Filter(ColumnRef(orders, "amount"), "gt", 0),),
        group_by=(ColumnRef(customers, "name"),),
        aggregations=(Aggregation("sum", ColumnRef(orders, "amount"), "revenue"),),
        calculations=(Calculation("ratio_percent", "revenue", "revenue", "share"),),
        sort=(Sort("share", "desc"),),
        limit=5,
        time_range=TimeRange(ColumnRef(orders, "order_date"), "2026-01-01", "2026-01-31"),
        evidence_requirements=(ColumnRef(orders, "order_id"),),
        output_type="table",
    )

    validate_query_plan(plan, CATALOG)


def test_invalid_plan_rejects_unknown_references_and_unsafe_limits() -> None:
    plan = QueryPlan(
        base_table="orders.csv::orders",
        selected_columns=(ColumnRef("orders.csv::orders", "missing"),),
        joins=(
            Join(
                ColumnRef("orders.csv::orders", "customer_id"),
                ColumnRef("orders.csv::orders; DROP TABLE orders", "id"),
            ),
        ),
        limit=100_001,
    )

    with pytest.raises(PlanValidationError) as error:
        validate_query_plan(plan, CATALOG)

    assert any("unknown table" in issue for issue in error.value.issues)
    assert any("unknown column" in issue for issue in error.value.issues)
    assert any("limit must be between" in issue for issue in error.value.issues)


def test_invalid_filters_aggregations_and_time_ranges_are_reported() -> None:
    orders = "orders.csv::orders"
    plan = QueryPlan(
        base_table=orders,
        filters=(
            Filter(ColumnRef(orders, "amount"), "between", [1]),
            Filter(ColumnRef(orders, "amount"), "gt", None),
        ),
        aggregations=(Aggregation("avg", None, "average"),),
        sort=(Sort("not_an_aggregate", "desc"),),
        time_range=TimeRange(ColumnRef(orders, "order_date"), "2026-02-01", "2026-01-01"),
    )

    with pytest.raises(PlanValidationError) as error:
        validate_query_plan(plan, CATALOG)

    assert any("between filter requires exactly two" in issue for issue in error.value.issues)
    assert any("requires a non-null value" in issue for issue in error.value.issues)
    assert any("avg requires a column" in issue for issue in error.value.issues)
    assert any("unknown column or aggregation alias" in issue for issue in error.value.issues)
    assert any("must not be after" in issue for issue in error.value.issues)


def test_join_must_be_connected_to_base_table() -> None:
    customers = "customers.xlsx::Customers"
    plan = QueryPlan(
        base_table=customers,
        joins=(
            Join(
                ColumnRef("orders.csv::orders", "customer_id"),
                ColumnRef("orders.csv::orders", "order_id"),
            ),
        ),
    )

    with pytest.raises(PlanValidationError, match="must connect"):
        validate_query_plan(plan, CATALOG)


def test_plan_rejects_business_rules_that_executor_cannot_apply() -> None:
    table = "orders.csv::orders"
    plan = QueryPlan(
        base_table=table,
        selected_columns=(ColumnRef(table, "order_id"),),
        business_rules=("Hủy đơn không tính doanh thu.",),
    )

    with pytest.raises(PlanValidationError, match="business_rules are not executable"):
        validate_query_plan(plan, CATALOG)


def test_plan_validates_scoped_aggregation_filters_and_dates() -> None:
    orders = "orders.csv::orders"
    plan = QueryPlan(
        base_table=orders,
        aggregations=(
            Aggregation(
                "count",
                None,
                "orders",
                filters=(
                    Filter(ColumnRef(orders, "missing_status"), "eq", "complete"),
                ),
                time_range=TimeRange(
                    ColumnRef(orders, "order_date"),
                    "2026-02-01",
                    "2026-01-01",
                ),
            ),
        ),
    )

    with pytest.raises(PlanValidationError) as error:
        validate_query_plan(plan, CATALOG)

    assert any(
        "unknown column orders.csv::orders.missing_status" in issue
        for issue in error.value.issues
    )
    assert any("start must not be after end" in issue for issue in error.value.issues)


def test_plan_rejects_unsafe_literals_non_iso_dates_and_boolean_limits() -> None:
    orders = "orders.csv::orders"
    plan = QueryPlan(
        base_table=orders,
        selected_columns=(ColumnRef(orders, "order_id"),),
        filters=(
            Filter(ColumnRef(orders, "amount"), "eq", object()),
            Filter(ColumnRef(orders, "amount"), "contains", 1),
        ),
        limit=True,
        time_range=TimeRange(ColumnRef(orders, "order_date"), "yesterday", "tomorrow"),
    )

    with pytest.raises(PlanValidationError) as error:
        validate_query_plan(plan, CATALOG)

    assert any("safe scalar" in issue for issue in error.value.issues)
    assert any("contains filter requires a text value" in issue for issue in error.value.issues)
    assert any("ISO-8601 dates" in issue for issue in error.value.issues)
    assert any("limit must be between" in issue for issue in error.value.issues)
