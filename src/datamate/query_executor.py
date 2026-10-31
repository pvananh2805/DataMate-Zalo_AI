"""Bounded deterministic query execution over streamed, typed table rows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import Iterable, Mapping, Sequence

from .query_plan import (
    Calculation,
    ColumnRef,
    Filter,
    Join,
    QueryPlan,
    validate_query_plan,
)

Row = Sequence[object]
TableRows = Iterable[Row]


class QueryExecutionError(RuntimeError):
    """Raised when a plan cannot be executed exactly within configured limits."""


@dataclass(frozen=True)
class ExecutionTrace:
    source_tables: tuple[str, ...]
    source_rows: Mapping[str, int]
    joined_rows: int
    filtered_rows: int
    group_count: int
    result_count: int


@dataclass(frozen=True)
class ExecutionResult:
    rows: tuple[dict[str, object], ...]
    trace: ExecutionTrace


@dataclass
class _AggregateState:
    count: int = 0
    total: Decimal = Decimal(0)
    extreme: object = None


def _column_key(ref: ColumnRef) -> str:
    return f"{ref.table}.{ref.column}"


def _row_value(
    context: Mapping[str, Row | None],
    column_indexes: Mapping[str, Mapping[str, int]],
    ref: ColumnRef,
) -> object:
    row = context.get(ref.table)
    if row is None:
        return None
    try:
        index = column_indexes[ref.table][ref.column]
    except KeyError as exc:
        raise QueryExecutionError(
            f"Column is unavailable in source schema: {ref.table}.{ref.column}"
        ) from exc
    return row[index]


def _matches(value: object, condition: Filter) -> bool:
    op = condition.operator
    expected = condition.value
    if op == "is_null":
        return value is None
    if op == "not_null":
        return value is not None
    if value is None:
        return False
    if op == "eq":
        return value == expected
    if op == "ne":
        return value != expected
    if op == "in":
        try:
            return value in expected
        except TypeError as exc:
            raise QueryExecutionError("in filter contains unsupported values") from exc
    if op == "contains":
        if not isinstance(value, str) or not isinstance(expected, str):
            raise QueryExecutionError("contains requires text on both sides")
        return expected.casefold() in value.casefold()
    try:
        if op == "between":
            low, high = expected
            return low <= value <= high
        if op == "lt":
            return value < expected
        if op == "lte":
            return value <= expected
        if op == "gt":
            return value > expected
        if op == "gte":
            return value >= expected
    except TypeError as exc:
        raise QueryExecutionError(
            f"Incompatible filter value types for operator {op}"
        ) from exc
    raise QueryExecutionError(f"Unsupported filter operator: {op}")


def _time_matches(value: object, plan: QueryPlan) -> bool:
    time_range = plan.time_range
    if time_range is None:
        return True
    if value is None:
        return False
    try:
        if time_range.inclusive_end:
            return time_range.start <= value <= time_range.end
        return time_range.start <= value < time_range.end
    except TypeError as exc:
        raise QueryExecutionError("Incompatible time-range and source value types") from exc


def _to_decimal(value: object, *, alias: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise QueryExecutionError(f"Aggregation {alias!r} requires numeric values")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise QueryExecutionError(f"Invalid numeric value for {alias!r}") from exc
    if not number.is_finite():
        raise QueryExecutionError(f"Non-finite numeric value for {alias!r}")
    return number


def _addition_precision(left: Decimal, right: Decimal) -> int:
    lowest_exponent = min(left.as_tuple().exponent, right.as_tuple().exponent)
    highest_digit = max(left.adjusted(), right.adjusted())
    return max(28, highest_digit - lowest_exponent + 2)


def _exact_add(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as decimal_context:
        decimal_context.prec = max(
            decimal_context.prec, _addition_precision(left, right)
        )
        return left + right


def _exact_subtract(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as decimal_context:
        decimal_context.prec = max(
            decimal_context.prec, _addition_precision(left, right)
        )
        return left - right


def _aggregate(
    rows: Iterable[Mapping[str, Row | None]],
    plan: QueryPlan,
    column_indexes: Mapping[str, Mapping[str, int]],
    *,
    max_groups: int,
) -> tuple[list[dict[str, object]], int]:
    grouped: dict[tuple[object, ...], tuple[tuple[object, ...], list[_AggregateState]]] = {}
    group_refs = plan.group_by

    for context in rows:
        key = tuple(_row_value(context, column_indexes, ref) for ref in group_refs)
        try:
            if key not in grouped:
                if len(grouped) >= max_groups:
                    raise QueryExecutionError(
                        f"Group count exceeds configured limit {max_groups}"
                    )
                grouped[key] = (
                    key,
                    [_AggregateState() for _ in plan.aggregations],
                )
        except TypeError as exc:
            raise QueryExecutionError("Group-by values must be hashable scalars") from exc
        _, states = grouped[key]
        for aggregation, state in zip(plan.aggregations, states, strict=True):
            value = (
                1
                if aggregation.column is None
                else _row_value(context, column_indexes, aggregation.column)
            )
            if aggregation.function == "count":
                if aggregation.column is None or value is not None:
                    state.count += 1
            elif value is not None:
                if aggregation.function in {"sum", "avg"}:
                    number = _to_decimal(value, alias=aggregation.alias)
                    state.total = _exact_add(state.total, number)
                    state.count += 1
                else:
                    try:
                        if state.count == 0:
                            state.extreme = value
                        elif aggregation.function == "min":
                            state.extreme = min(state.extreme, value)
                        else:
                            state.extreme = max(state.extreme, value)
                    except TypeError as exc:
                        raise QueryExecutionError(
                            f"Mixed values cannot be compared for {aggregation.alias!r}"
                        ) from exc
                    state.count += 1

    if not grouped and not group_refs:
        grouped[()] = ((), [_AggregateState() for _ in plan.aggregations])

    output: list[dict[str, object]] = []
    for key, states in grouped.values():
        result = {
            _column_key(ref): value for ref, value in zip(group_refs, key, strict=True)
        }
        for aggregation, state in zip(plan.aggregations, states, strict=True):
            if aggregation.function == "count":
                value: object = state.count
            elif state.count == 0:
                value = None
            elif aggregation.function == "sum":
                value = state.total
            elif aggregation.function == "avg":
                with localcontext() as decimal_context:
                    decimal_context.prec = max(decimal_context.prec, 50)
                    value = state.total / state.count
            else:
                value = state.extreme
            result[aggregation.alias] = value
        output.append(result)
    return output, len(grouped)


def _calculate(
    rows: list[dict[str, object]], calculations: Sequence[Calculation]
) -> None:
    for row in rows:
        for calculation in calculations:
            left = row.get(calculation.left)
            right = row.get(calculation.right)
            if left is None or right is None:
                row[calculation.alias] = None
                continue
            left_number = _to_decimal(left, alias=calculation.left)
            right_number = _to_decimal(right, alias=calculation.right)
            if calculation.function == "difference":
                row[calculation.alias] = _exact_subtract(left_number, right_number)
                continue
            if right_number == 0:
                raise QueryExecutionError(
                    f"Calculation {calculation.alias!r} divides by zero"
                )
            numerator = (
                _exact_subtract(left_number, right_number)
                if calculation.function == "percent_change"
                else left_number
            )
            multiplier = (
                Decimal(100)
                if calculation.function in {"ratio_percent", "percent_change"}
                else Decimal(1)
            )
            with localcontext() as decimal_context:
                decimal_context.prec = max(decimal_context.prec, 50)
                row[calculation.alias] = numerator / right_number * multiplier


def execute_query_plan(
    plan: QueryPlan,
    *,
    catalog: Mapping[str, Sequence[str]],
    sources: Mapping[str, TableRows],
    max_join_rows: int = 100_000,
    max_groups: int = 100_000,
    max_result_rows: int = 10_000,
) -> ExecutionResult:
    """Run allow-listed operations over streamed rows without executing SQL/code."""
    for name, value in (
        ("max_join_rows", max_join_rows),
        ("max_groups", max_groups),
        ("max_result_rows", max_result_rows),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    schema_sets: dict[str, set[str]] = {}
    schemas: dict[str, tuple[str, ...]] = {}
    column_indexes: dict[str, dict[str, int]] = {}
    for table, columns in catalog.items():
        column_tuple = tuple(columns)
        if len(set(column_tuple)) != len(column_tuple):
            raise QueryExecutionError(
                f"Duplicate column names are ambiguous for {table}"
            )
        schemas[table] = column_tuple
        schema_sets[table] = set(column_tuple)
        column_indexes[table] = {
            column: index for index, column in enumerate(column_tuple)
        }
    validate_query_plan(plan, schema_sets)

    involved_tables = {plan.base_table}
    for join in plan.joins:
        involved_tables.update((join.left.table, join.right.table))
    missing_sources = involved_tables - sources.keys()
    if missing_sources:
        raise QueryExecutionError(
            f"Missing row sources for tables: {', '.join(sorted(missing_sources))}"
        )

    joined_tables = {plan.base_table}
    join_indexes: dict[str, dict[object, list[Row]]] = {}
    source_row_counts: dict[str, int] = {}
    for join in plan.joins:
        new_table = (
            join.right.table
            if join.right.table not in joined_tables
            else join.left.table
        )
        new_key = join.right if join.right.table == new_table else join.left
        schema = schemas[new_table]
        try:
            key_index = column_indexes[new_table][new_key.column]
        except KeyError as exc:
            raise QueryExecutionError(
                f"Join key is unavailable: {new_table}.{new_key.column}"
            ) from exc
        index: dict[object, list[Row]] = defaultdict(list)
        row_count = 0
        for row_count, raw_row in enumerate(sources[new_table], start=1):
            row = tuple(raw_row)
            if len(row) != len(schema):
                raise QueryExecutionError(
                    f"Row width mismatch in {new_table} at row {row_count}"
                )
            key = row[key_index]
            if key is not None:
                try:
                    index[key].append(row)
                except TypeError as exc:
                    raise QueryExecutionError(
                        f"Join key values must be hashable scalars: {new_table}"
                    ) from exc
            if row_count > max_join_rows:
                raise QueryExecutionError(
                    f"Join input {new_table} exceeds configured limit {max_join_rows}"
                )
        source_row_counts[new_table] = row_count
        join_indexes[new_table] = index
        joined_tables.add(new_table)

    joined_row_count = 0

    def contexts() -> Iterable[dict[str, Row | None]]:
        nonlocal joined_row_count
        base_row_count = 0
        for base_row_count, raw_row in enumerate(sources[plan.base_table], start=1):
            row = tuple(raw_row)
            if len(row) != len(schemas[plan.base_table]):
                raise QueryExecutionError(
                    f"Row width mismatch in {plan.base_table} at row {base_row_count}"
                )
            current: list[dict[str, Row | None]] = [{plan.base_table: row}]
            for join in plan.joins:
                new_table = (
                    join.right.table
                    if join.right.table not in current[0]
                    else join.left.table
                )
                probe_key = join.left if join.left.table != new_table else join.right
                expanded: list[dict[str, Row | None]] = []
                for context in current:
                    key = _row_value(context, column_indexes, probe_key)
                    matches = join_indexes[new_table].get(key, []) if key is not None else []
                    if matches:
                        for match in matches:
                            expanded.append({**context, new_table: match})
                    elif join.kind == "left":
                        expanded.append({**context, new_table: None})
                current = expanded
                if not current:
                    break
                if len(current) + joined_row_count > max_result_rows:
                    raise QueryExecutionError(
                        f"Intermediate joined rows exceed configured limit {max_result_rows}"
                    )
            for context in current:
                joined_row_count += 1
                yield context
        source_row_counts[plan.base_table] = base_row_count

    def filtered_contexts() -> Iterable[dict[str, Row | None]]:
        for context in contexts():
            if not all(
                _matches(_row_value(context, column_indexes, item.column), item)
                for item in plan.filters
            ):
                continue
            if plan.time_range and not _time_matches(
                _row_value(context, column_indexes, plan.time_range.column), plan
            ):
                continue
            yield context

    filtered_count = 0

    def counted_contexts() -> Iterable[dict[str, Row | None]]:
        nonlocal filtered_count
        for context in filtered_contexts():
            filtered_count += 1
            yield context

    if plan.aggregations or plan.calculations or plan.group_by:
        rows, group_count = _aggregate(
            counted_contexts(), plan, column_indexes, max_groups=max_groups
        )
        _calculate(rows, plan.calculations)
    else:
        rows = []
        group_count = 0
        for context in counted_contexts():
            row = {
                _column_key(ref): _row_value(context, column_indexes, ref)
                for ref in plan.selected_columns
            }
            rows.append(row)
            if len(rows) > max_result_rows:
                raise QueryExecutionError(
                    f"Result row count exceeds configured limit {max_result_rows}"
                )

    for ordering in reversed(plan.sort):
        key = _column_key(ordering.key) if isinstance(ordering.key, ColumnRef) else ordering.key
        non_null = [row for row in rows if row[key] is not None]
        null_rows = [row for row in rows if row[key] is None]
        try:
            non_null.sort(
                key=lambda result: result[key],
                reverse=ordering.direction == "desc",
            )
        except TypeError as exc:
            raise QueryExecutionError(f"Values for sort key {key!r} are incomparable") from exc
        rows = non_null + null_rows
    if plan.limit is not None:
        rows = rows[: plan.limit]
    if len(rows) > max_result_rows:
        raise QueryExecutionError(
            f"Result row count exceeds configured limit {max_result_rows}"
        )
    return ExecutionResult(
        rows=tuple(rows),
        trace=ExecutionTrace(
            source_tables=tuple(sorted(involved_tables)),
            source_rows=dict(source_row_counts),
            joined_rows=joined_row_count,
            filtered_rows=filtered_count,
            group_count=group_count,
            result_count=len(rows),
        ),
    )
