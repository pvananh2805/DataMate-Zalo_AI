"""Chạy suy luận offline và ghi nhật ký nội bộ dạng JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datamate.answer_validation import AnswerValidationError
from datamate.business_context import (
    BusinessContextDocuments,
    load_business_context,
)
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
from datamate.submission import SubmissionError, SubmissionWriter


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


def _process_questions(
    input_path: Path,
    business_root: Path,
    planner: JsonPlanAdapter,
    log_path: Path,
    coercions: dict[str, dict[ColumnRef, ColumnCoercion]],
    *,
    rulebook_path: Path | None,
    calendar_path: Path | None,
    submission_writer: SubmissionWriter | None,
) -> tuple[int, int]:
    succeeded = 0
    failed = 0
    context_cache: dict[str, BusinessContextDocuments] = {}
    with input_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                print(f"Bỏ qua dòng câu hỏi trống {line_number}", file=sys.stderr)
                failed += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"JSON không hợp lệ ở dòng {line_number}: {exc}", file=sys.stderr)
                failed += 1
                continue
            if not isinstance(record, dict):
                print(f"Dòng {line_number} phải là JSON object", file=sys.stderr)
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
                print(
                    f"Thiếu id/business_id/question ở dòng {line_number}",
                    file=sys.stderr,
                )
                failed += 1
                continue
            if as_of_date is not None and not isinstance(as_of_date, str):
                print(f"as_of_date không hợp lệ ở dòng {line_number}", file=sys.stderr)
                failed += 1
                continue
            try:
                business_dir = _safe_business_directory(business_root, business_id)
                if business_id not in context_cache:
                    context_cache[business_id] = load_business_context(
                        business_dir,
                        business_root,
                        rulebook_path=rulebook_path,
                        calendar_path=calendar_path,
                    )
                    missing_sources = [
                        name
                        for name, metadata in context_cache[
                            business_id
                        ].source_metadata.items()
                        if metadata["status"] == "missing"
                    ]
                    if missing_sources:
                        print(
                            f"Cảnh báo: Thiếu tài liệu ngữ cảnh cho {business_id}: "
                            f"{', '.join(missing_sources)}",
                            file=sys.stderr,
                        )
                documents = context_cache[business_id]
                result = execute_question_and_log(
                    business_dir,
                    question,
                    planner,
                    question_id=question_id,
                    business_id=business_id,
                    as_of_date=as_of_date,
                    log_path=log_path,
                    input_root=business_root,
                    input_metadata={
                        "source_file": input_path.name,
                        "line_number": line_number,
                        "business_context_sources": documents.source_metadata,
                    },
                    column_coercions=coercions.get(business_id),
                    rulebook=documents.rulebook,
                    business_readme=documents.business_readme,
                    calendar_columns=documents.calendar_columns,
                    calendar_rows=documents.calendar_rows,
                )
                if submission_writer is not None:
                    submission_writer.append(
                        question_id,
                        result.answer,
                        plan=result.plan,
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
                    f"Câu hỏi {question_id} lỗi: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                failed += 1
    return succeeded, failed


def main() -> None:
    for output in (sys.stdout, sys.stderr):
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Tệp câu hỏi JSONL")
    parser.add_argument(
        "--business-data-root",
        required=True,
        help="Thư mục chứa thư mục con cho từng business_id",
    )
    parser.add_argument("--model-path", required=True, help="Checkpoint mô hình cục bộ")
    parser.add_argument("--log", required=True, help="Tệp nhật ký JSONL nội bộ")
    parser.add_argument("--coercions", type=Path, help="JSON policy chuẩn hóa đã duyệt")
    parser.add_argument("--rulebook", type=Path, help="Markdown rulebook dùng chung")
    parser.add_argument("--calendar", type=Path, help="CSV calendar dùng chung")
    parser.add_argument(
        "--submission-output",
        type=Path,
        help="JSONL tham khảo rulebook training; chưa xác minh với validator cuộc thi",
    )
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--parameter-cap", type=int, default=15_000_000_000)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    business_root = Path(args.business_data_root).resolve()
    log_path = Path(args.log).resolve()
    submission_path = (
        Path(args.submission_output).resolve()
        if args.submission_output is not None
        else None
    )
    if not input_path.is_file() or not business_root.is_dir():
        parser.error("input file and business-data-root directory must exist")
    if submission_path is not None and submission_path.exists():
        parser.error(f"Tệp submission đã tồn tại: {submission_path}")
    try:
        validate_log_destination(
            log_path,
            forbidden_roots=(input_path.parent, business_root),
        )
        if submission_path is not None:
            validate_log_destination(
                submission_path,
                forbidden_roots=(input_path.parent, business_root),
            )
            if submission_path == log_path:
                raise ValueError("Submission và run log phải là hai tệp khác nhau")
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
    submission_writer = (
        SubmissionWriter(submission_path)
        if submission_path is not None
        else None
    )
    try:
        succeeded, failed = _process_questions(
            input_path,
            business_root,
            planner,
            log_path,
            coercions,
            rulebook_path=args.rulebook,
            calendar_path=args.calendar,
            submission_writer=submission_writer,
        )
        if submission_writer is not None:
            if failed:
                submission_writer.abort()
            else:
                submission_writer.commit()
    finally:
        if submission_writer is not None:
            submission_writer.abort()
    if submission_writer is not None and failed:
        print(
            "Không tạo submission do có câu hỏi lỗi; tệp tạm đã được loại bỏ.",
            file=sys.stderr,
        )
    print(
        json.dumps(
            {
                "processed": succeeded + failed,
                "succeeded": succeeded,
                "failed": failed,
                "run_log": str(log_path),
                "metric_status": "internal_run_log_not_submission",
                "submission": (
                    str(submission_path)
                    if submission_writer is not None and failed == 0
                    else None
                ),
                "submission_schema_status": (
                    "training_rulebook_reference_not_platform_verified"
                    if submission_writer is not None and failed == 0
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
