"""Connect business data discovery, plan validation, execution, and answer checks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .answer_validation import (
    AnswerValidationError,
    Evidence,
    ValidatedAnswer,
    validate_answer,
)
from .data_loader import BusinessDataLoader, DataLoadError, TableRef
from .query_executor import ExecutionResult, QueryExecutionError, execute_query_plan
from .query_plan import ColumnRef, PlanValidationError, QueryPlan
from .run_logging import RunRecord, append_run_record, validate_log_destination


class PipelineError(RuntimeError):
    """Raised when an execution result cannot be mapped to a typed answer."""


@dataclass(frozen=True)
class BusinessSchema:
    tables: dict[str, TableRef]
    columns: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class PipelineResult:
    answer: ValidatedAnswer
    execution: ExecutionResult


def table_identifier(table: TableRef) -> str:
    """Return a stable table key unique within one business directory."""
    discriminator = table.sheet_name or table.sqlite_table or table.name
    return f"{table.relative_path}::{discriminator}"


def discover_business_schema(business_dir: Path | str) -> BusinessSchema:
    loader = BusinessDataLoader(business_dir)
    tables = {table_identifier(table): table for table in loader.discover()}
    return BusinessSchema(
        tables=tables,
        columns={identifier: table.columns for identifier, table in tables.items()},
    )


def _answer_value(result: ExecutionResult, answer_type: str) -> object:
    if answer_type == "table":
        return list(result.rows)
    if answer_type == "list":
        if any(len(row) != 1 for row in result.rows):
            raise PipelineError("list output requires exactly one result column")
        return [next(iter(row.values())) for row in result.rows]
    if len(result.rows) != 1 or len(result.rows[0]) != 1:
        raise PipelineError(
            f"{answer_type} output requires exactly one scalar result"
        )
    return next(iter(result.rows[0].values()))


def _evidence_for(plan: QueryPlan, execution: ExecutionResult) -> tuple[Evidence, ...]:
    references: set[ColumnRef] = {
        *plan.selected_columns,
        *plan.group_by,
        *plan.evidence_requirements,
    }
    references.update(
        aggregation.column
        for aggregation in plan.aggregations
        if aggregation.column is not None
    )
    references.update(item.column for item in plan.filters)
    if plan.time_range is not None:
        references.add(plan.time_range.column)
    for join in plan.joins:
        references.update((join.left, join.right))

    evidence: list[Evidence] = []
    for table in execution.trace.source_tables:
        columns = tuple(
            sorted(ref.column for ref in references if ref.table == table)
        )
        evidence.append(
            Evidence(
                source_table=table,
                columns=columns,
                operation="validated_query_plan",
                row_count=execution.trace.source_rows.get(table, 0),
                filter_summary=f"{len(plan.filters)} filters; "
                f"time_range={'yes' if plan.time_range else 'no'}",
            )
        )
    return tuple(evidence)


def execute_business_plan(
    business_dir: Path | str,
    plan: QueryPlan,
    *,
    max_join_rows: int = 100_000,
    max_groups: int = 100_000,
    max_result_rows: int = 10_000,
) -> PipelineResult:
    """Discover the local schema and execute a plan using only approved readers."""
    loader = BusinessDataLoader(business_dir)
    discovered = loader.discover()
    tables = {table_identifier(table): table for table in discovered}
    if plan.output_type in {"unanswerable", "abstain"}:
        raise PipelineError(
            "Non-answer statuses must be returned through an explicit abstention path"
        )
    if plan.output_type is None:
        raise PipelineError("Query plan does not declare an output type")
    involved_tables = {plan.base_table}
    for join in plan.joins:
        involved_tables.update((join.left.table, join.right.table))
    catalog = {
        identifier: tables[identifier].columns
        for identifier in involved_tables
        if identifier in tables
    }
    sources = {
        identifier: loader.iter_rows(tables[identifier])
        for identifier in involved_tables
        if identifier in tables
    }
    execution = execute_query_plan(
        plan,
        catalog=catalog,
        sources=sources,
        max_join_rows=max_join_rows,
        max_groups=max_groups,
        max_result_rows=max_result_rows,
    )

    if execution.trace.filtered_rows == 0:
        has_scalar_count = (
            plan.output_type == "number"
            and not plan.group_by
            and any(item.function == "count" for item in plan.aggregations)
        )
        value = (
            _answer_value(execution, plan.output_type)
            if has_scalar_count or plan.output_type in {"list", "table"}
            else None
        )
    else:
        value = _answer_value(execution, plan.output_type)
    if value is None and execution.trace.filtered_rows == 0:
        answer = ValidatedAnswer(
            answer_type="unanswerable",
            value=None,
            status="no_data",
            explanation="Không có dòng dữ liệu phù hợp với điều kiện truy vấn.",
            evidence=_evidence_for(plan, execution),
        )
    else:
        value = _answer_value(execution, plan.output_type)
        if value is None:
            answer = ValidatedAnswer(
                answer_type="unanswerable",
                value=None,
                status="insufficient_data",
                explanation="Dữ liệu phù hợp không có giá trị cần thiết để tính.",
                evidence=_evidence_for(plan, execution),
            )
        else:
            answer = ValidatedAnswer(
                answer_type=plan.output_type,
                value=value,
                status="answered",
                evidence=_evidence_for(plan, execution),
            )
    validate_answer(answer)
    return PipelineResult(answer=answer, execution=execution)


def execute_and_log_business_plan(
    business_dir: Path | str,
    plan: QueryPlan,
    *,
    question_id: str,
    business_id: str,
    question: str,
    model: str,
    log_path: Path | str,
    input_root: Path | str,
    input_metadata: dict[str, object] | None = None,
) -> PipelineResult:
    """Execute one prepared plan and append success or known failure to JSONL."""
    validate_log_destination(log_path, forbidden_roots=(input_root,))
    started = time.perf_counter()
    try:
        result = execute_business_plan(business_dir, plan)
    except (
        DataLoadError,
        PlanValidationError,
        QueryExecutionError,
        PipelineError,
        AnswerValidationError,
    ) as exc:
        append_run_record(
            log_path,
            RunRecord(
                question_id=question_id,
                business_id=business_id,
                question=question,
                model=model,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                plan=plan,
                error=f"{type(exc).__name__}: {exc}",
                input_metadata=input_metadata,
            ),
            forbidden_roots=(input_root,),
        )
        raise

    append_run_record(
        log_path,
        RunRecord(
            question_id=question_id,
            business_id=business_id,
            question=question,
            model=model,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            plan=plan,
            execution=result.execution.trace,
            answer=result.answer,
            input_metadata=input_metadata,
        ),
        forbidden_roots=(input_root,),
    )
    return result
