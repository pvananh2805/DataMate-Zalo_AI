"""Run offline question-to-answer inference and write internal JSONL run logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datamate.answer_validation import AnswerValidationError
from datamate.data_loader import DataLoadError
from datamate.local_inference import LocalTransformersBackend
from datamate.normalization import ColumnCoercion, ValueNormalizationError
from datamate.pipeline import (
    PipelineError,
    execute_question_and_log,
)
from datamate.query_executor import QueryExecutionError
from datamate.query_plan import PlanValidationError, ColumnRef
from datamate.question_understanding import (
    JsonPlanAdapter,
    QuestionUnderstandingError,
)
from datamate.run_logging import validate_log_destination


def _read_coercions(path: Path | None) -> dict[str, dict[ColumnRef, ColumnCoercion]]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read coercion configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Coercion configuration must be a JSON object")
    result: dict[str, dict[ColumnRef, ColumnCoercion]] = {}
    for business_id, table_policies in payload.items():
        if not isinstance(business_id, str) or not isinstance(table_policies, dict):
            raise ValueError("Coercions must be grouped by business and table")
        policies: dict[ColumnRef, ColumnCoercion] = {}
        for table_id, columns in table_policies.items():
            if not isinstance(table_id, str) or not isinstance(columns, dict):
                raise ValueError(f"Invalid table coercions for {business_id}")
            for column, policy in columns.items():
                if not isinstance(column, str) or not isinstance(policy, dict):
                    raise ValueError(f"Invalid column coercion for {table_id}")
                try:
                    policies[ColumnRef(table_id, column)] = ColumnCoercion(**policy)
                except TypeError as exc:
                    raise ValueError(
                        f"Invalid coercion policy for {table_id}.{column}: {exc}"
                    ) from exc
        result[business_id] = policies
    return result


def _safe_business_directory(root: Path, business_id: object) -> Path:
    if not isinstance(business_id, str) or not business_id:
        raise ValueError("business_id must be non-empty text")
    if Path(business_id).name != business_id or business_id in {".", ".."}:
        raise ValueError(f"Unsafe business_id: {business_id!r}")
    business_dir = (root / business_id).resolve()
    if not business_dir.is_relative_to(root):
        raise ValueError(f"Unsafe business directory: {business_id}")
    return business_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Questions JSONL")
    parser.add_argument(
        "--business-data-root",
        required=True,
        help="Directory containing one subdirectory per business_id",
    )
    parser.add_argument("--model-path", required=True, help="Local model checkpoint")
    parser.add_argument("--log", required=True, help="New or appendable internal JSONL log")
    parser.add_argument("--coercions", type=Path, help="Reviewed coercion policy JSON")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--parameter-cap", type=int, default=15_000_000_000)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    business_root = Path(args.business_data_root).resolve()
    log_path = Path(args.log).resolve()
    if not input_path.is_file() or not business_root.is_dir():
        parser.error("input file and business-data-root directory must exist")
    try:
        validate_log_destination(
            log_path,
            forbidden_roots=(input_path.parent, business_root),
        )
        coercions = _read_coercions(args.coercions)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    backend = LocalTransformersBackend(
        args.model_path,
        precision=args.precision,
        max_new_tokens=args.max_new_tokens,
        parameter_cap=args.parameter_cap,
    )
    try:
        backend.load()
    except QuestionUnderstandingError as exc:
        parser.error(str(exc))
    planner = JsonPlanAdapter(backend, model_name=backend.model_name)
    succeeded = 0
    failed = 0
    with input_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                print(f"Skipped blank input line {line_number}", file=sys.stderr)
                failed += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON at line {line_number}: {exc}", file=sys.stderr)
                failed += 1
                continue
            if not isinstance(record, dict):
                print(f"Expected JSON object at line {line_number}", file=sys.stderr)
                failed += 1
                continue
            question_id = record.get("id")
            business_id = record.get("business_id")
            question = record.get("question")
            as_of_date = record.get("as_of_date")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (question_id, business_id, question)
            ):
                print(f"Missing id/business_id/question at line {line_number}", file=sys.stderr)
                failed += 1
                continue
            if as_of_date is not None and not isinstance(as_of_date, str):
                print(f"Invalid as_of_date at line {line_number}", file=sys.stderr)
                failed += 1
                continue
            try:
                business_dir = _safe_business_directory(business_root, business_id)
                execute_question_and_log(
                    business_dir,
                    question,
                    planner,
                    question_id=question_id,
                    business_id=business_id,
                    as_of_date=as_of_date,
                    log_path=log_path,
                    input_root=business_root,
                    input_metadata={"source_file": input_path.name, "line_number": line_number},
                    column_coercions=coercions.get(business_id),
                )
                succeeded += 1
            except (
                DataLoadError,
                PlanValidationError,
                QueryExecutionError,
                ValueNormalizationError,
                PipelineError,
                AnswerValidationError,
                QuestionUnderstandingError,
                ValueError,
            ) as exc:
                print(
                    f"Question {question_id} failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                failed += 1
    print(
        json.dumps(
            {
                "processed": succeeded + failed,
                "succeeded": succeeded,
                "failed": failed,
                "run_log": str(log_path),
                "metric_status": "internal_run_log_not_submission",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
