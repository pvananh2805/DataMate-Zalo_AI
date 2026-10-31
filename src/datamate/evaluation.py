"""Strict local evaluation; this is not the competition's Penalized Accuracy."""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TextIO


class EvaluationError(ValueError):
    """Raised when an evaluation input or output cannot be validated."""


def _validate_nested_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int, date, datetime)):
        return True
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_validate_nested_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _validate_nested_value(item)
            for key, item in value.items()
        )
    return False


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Non-finite decimal")
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _jsonl_records(stream: TextIO, source: Path):
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            raise EvaluationError(f"Blank JSONL line at {source}:{line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(
                f"Invalid JSON at {source}:{line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise EvaluationError(
                f"Expected a JSON object at {source}:{line_number}"
            )
        yield line_number, record


def _canonical_value(answer_type: object, value: object) -> str:
    if not isinstance(answer_type, str):
        raise EvaluationError("answer type must be a string")
    if answer_type in {"number", "percent"}:
        if isinstance(value, bool) or value is None:
            raise EvaluationError(f"{answer_type} answer must be numeric")
        try:
            number = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise EvaluationError(f"{answer_type} answer is not numeric") from exc
        if not number.is_finite():
            raise EvaluationError(f"{answer_type} answer must be finite")
        return str(number.normalize())
    if answer_type == "date":
        if not isinstance(value, str):
            raise EvaluationError("date answer must be an ISO-8601 string")
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise EvaluationError("date answer must be an ISO-8601 string") from exc
    if answer_type == "string":
        if not isinstance(value, str):
            raise EvaluationError("string answer must be text")
        return json.dumps(value, ensure_ascii=False)
    if answer_type == "boolean":
        if not isinstance(value, bool):
            raise EvaluationError("boolean answer must be true or false")
    elif answer_type == "unanswerable":
        if value is not None:
            raise EvaluationError("unanswerable answer must be null")
    elif answer_type == "table":
        if not isinstance(value, (dict, list)):
            raise EvaluationError("table answer must be an object or array")
    elif answer_type == "list":
        if not isinstance(value, list):
            raise EvaluationError("list answer must be an array")
    elif answer_type not in {"boolean", "unanswerable", "table", "list"}:
        raise EvaluationError(f"unsupported answer type: {answer_type}")
    try:
        if not _validate_nested_value(value):
            raise EvaluationError(f"{answer_type} answer contains invalid JSON values")
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{answer_type} answer is not valid JSON data") from exc


def _required_id(record: dict[str, object], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{context}: {key} must be a non-empty string")
    return value


def _load_questions(
    connection: sqlite3.Connection,
    source: Path,
) -> None:
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, record in _jsonl_records(stream, source):
            context = f"{source}:{line_number}"
            question_id = _required_id(record, "id", context)
            business_id = _required_id(record, "business_id", context)
            try:
                connection.execute(
                    "INSERT INTO question_business VALUES (?, ?)",
                    (question_id, business_id),
                )
            except sqlite3.IntegrityError as exc:
                raise EvaluationError(
                    f"{context}: duplicate question id {question_id!r}"
                ) from exc


def _load_gold(connection: sqlite3.Connection, source: Path) -> int:
    count = 0
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, record in _jsonl_records(stream, source):
            context = f"{source}:{line_number}"
            question_id = _required_id(record, "id", context)
            answer_type = record.get("type")
            answer_value = _canonical_value(answer_type, record.get("answer"))
            try:
                connection.execute(
                    "INSERT INTO gold VALUES (?, ?, ?)",
                    (question_id, answer_type, answer_value),
                )
            except sqlite3.IntegrityError as exc:
                raise EvaluationError(
                    f"{context}: duplicate gold question id {question_id!r}"
                ) from exc
            count += 1
    if count == 0:
        raise EvaluationError("Gold answer file is empty")
    return count


def _load_predictions(connection: sqlite3.Connection, source: Path) -> int:
    count = 0
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, record in _jsonl_records(stream, source):
            context = f"{source}:{line_number}"
            question_id = _required_id(
                record,
                "question_id" if "question_id" in record else "id",
                context,
            )
            nested = record.get("answer")
            answer_record = nested if isinstance(nested, dict) else {}
            answer_type = (
                record.get("type")
                or answer_record.get("answer_type")
                or record.get("answer_type")
            )
            if isinstance(nested, dict) and "value" in nested:
                value = nested["value"]
            else:
                value = nested
            valid = 1
            try:
                canonical = _canonical_value(answer_type, value)
            except EvaluationError:
                canonical = "null"
                valid = 0
            status = answer_record.get("status", record.get("status"))
            business_id = record.get("business_id")
            if not isinstance(business_id, str):
                business_id = None
            elapsed_ms = record.get("elapsed_ms")
            if (
                isinstance(elapsed_ms, bool)
                or not isinstance(elapsed_ms, (int, float))
                or not math.isfinite(elapsed_ms)
                or elapsed_ms < 0
            ):
                elapsed_ms = None
            try:
                connection.execute(
                    "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        question_id,
                        answer_type if isinstance(answer_type, str) else None,
                        canonical,
                        status if isinstance(status, str) else None,
                        business_id,
                        elapsed_ms,
                        valid,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise EvaluationError(
                    f"{context}: duplicate prediction question id {question_id!r}"
                ) from exc
            count += 1
    return count


def evaluate_predictions(
    answers_path: Path | str,
    predictions_path: Path | str,
    report_path: Path | str,
    *,
    questions_path: Path | str | None = None,
) -> dict[str, object]:
    """Compute transparent exact-match diagnostics without applying official PA."""
    answers = Path(answers_path).resolve()
    predictions = Path(predictions_path).resolve()
    destination = Path(report_path).resolve()
    questions = Path(questions_path).resolve() if questions_path else None
    if not answers.is_file() or not predictions.is_file():
        raise EvaluationError("Gold answers and predictions must exist")
    if questions is not None and not questions.is_file():
        raise EvaluationError("Questions file does not exist")
    sources = (answers, predictions) + ((questions,) if questions else ())
    if destination.exists():
        raise EvaluationError(f"Report already exists: {destination}")
    if any(
        destination == source.parent or destination.is_relative_to(source.parent)
        for source in sources
    ):
        raise EvaluationError("Report must be outside input directories")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="datamate-eval-") as temporary:
        connection = sqlite3.connect(Path(temporary) / "evaluation.sqlite")
        try:
            connection.execute(
                "CREATE TABLE gold (question_id TEXT PRIMARY KEY, "
                "answer_type TEXT NOT NULL, answer_value TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE predictions (question_id TEXT PRIMARY KEY, "
                "answer_type TEXT, answer_value TEXT NOT NULL, status TEXT, "
                "business_id TEXT, elapsed_ms REAL, valid INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE question_business ("
                "question_id TEXT PRIMARY KEY, business_id TEXT NOT NULL)"
            )
            if questions is not None:
                _load_questions(connection, questions)
            gold_count = _load_gold(connection, answers)
            prediction_count = _load_predictions(connection, predictions)
            connection.commit()
            report = _aggregate_report(
                connection,
                gold_count=gold_count,
                prediction_count=prediction_count,
            )
        finally:
            connection.close()

    destination.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _aggregate_report(
    connection: sqlite3.Connection,
    *,
    gold_count: int,
    prediction_count: int,
) -> dict[str, object]:
    exact_matches = 0
    type_matches = 0
    missing_predictions = 0
    invalid_predictions = 0
    business_mismatches = 0
    unanswerable_total = 0
    unanswerable_correct = 0
    latency_total = 0.0
    latency_count = 0
    max_latency = 0.0
    by_type: dict[str, dict[str, int]] = {}
    by_business: dict[str, dict[str, int]] = {}
    query = """
        SELECT g.question_id, g.answer_type, g.answer_value,
               p.answer_type, p.answer_value, p.business_id, p.elapsed_ms,
               p.valid, q.business_id
        FROM gold AS g
        LEFT JOIN predictions AS p ON p.question_id = g.question_id
        LEFT JOIN question_business AS q ON q.question_id = g.question_id
        ORDER BY g.question_id
    """
    for (
        question_id,
        gold_type,
        gold_value,
        predicted_type,
        predicted_value,
        predicted_business,
        elapsed_ms,
        valid,
        gold_business,
    ) in connection.execute(query):
        type_bucket = by_type.setdefault(
            gold_type,
            {"count": 0, "type_correct": 0, "exact_match": 0},
        )
        business_id = gold_business or predicted_business or "unknown"
        business_bucket = by_business.setdefault(
            business_id,
            {"count": 0, "exact_match": 0},
        )
        type_bucket["count"] += 1
        business_bucket["count"] += 1
        if predicted_type is None:
            missing_predictions += 1
        if valid == 0:
            invalid_predictions += 1
        type_matches_this = valid == 1 and predicted_type == gold_type
        exact_this = type_matches_this and _same_answer_value(
            gold_type,
            gold_value,
            predicted_value,
        )
        if type_matches_this:
            type_matches += 1
            type_bucket["type_correct"] += 1
        if exact_this:
            exact_matches += 1
            type_bucket["exact_match"] += 1
            business_bucket["exact_match"] += 1
        if gold_type == "unanswerable":
            unanswerable_total += 1
            if exact_this:
                unanswerable_correct += 1
        if (
            gold_business is not None
            and predicted_business is not None
            and gold_business != predicted_business
        ):
            business_mismatches += 1
        if elapsed_ms is not None:
            latency_total += elapsed_ms
            latency_count += 1
            max_latency = max(max_latency, elapsed_ms)

    extra_predictions = connection.execute(
        "SELECT COUNT(*) FROM predictions AS p "
        "LEFT JOIN gold AS g ON g.question_id = p.question_id "
        "WHERE g.question_id IS NULL"
    ).fetchone()[0]
    return {
        "metric": "local_strict_exact_match_not_official_PA",
        "official_pa_computed": False,
        "aliases_compared": False,
        "gold_count": gold_count,
        "prediction_count": prediction_count,
        "exact_matches": exact_matches,
        "exact_match_fraction": exact_matches / gold_count if gold_count else 0.0,
        "answer_type_matches": type_matches,
        "answer_type_accuracy": type_matches / gold_count if gold_count else 0.0,
        "missing_predictions": missing_predictions,
        "extra_predictions": extra_predictions,
        "invalid_predictions": invalid_predictions,
        "business_id_mismatches": business_mismatches,
        "unanswerable_count": unanswerable_total,
        "unanswerable_exact_matches": unanswerable_correct,
        "latency_ms": {
            "count": latency_count,
            "average": latency_total / latency_count if latency_count else None,
            "maximum": max_latency if latency_count else None,
        },
        "by_answer_type": by_type,
        "by_business": by_business,
    }


def _same_answer_value(
    answer_type: str,
    gold_value: str,
    predicted_value: str,
) -> bool:
    if answer_type not in {"list", "table"}:
        return gold_value == predicted_value
    try:
        gold = json.loads(gold_value)
        predicted = json.loads(predicted_value)
    except json.JSONDecodeError:
        return False
    return _structured_equal(gold, predicted)


def _structured_equal(gold: object, predicted: object) -> bool:
    if isinstance(gold, bool):
        return isinstance(predicted, bool) and gold == predicted
    if isinstance(gold, (int, float)):
        if isinstance(predicted, bool):
            return False
        try:
            expected_number = Decimal(str(gold))
            predicted_number = Decimal(str(predicted))
        except (InvalidOperation, ValueError):
            return False
        return (
            expected_number.is_finite()
            and predicted_number.is_finite()
            and expected_number == predicted_number
        )
    if gold is None:
        return predicted is None
    if isinstance(gold, str):
        return isinstance(predicted, str) and gold == predicted
    if isinstance(gold, list):
        return (
            isinstance(predicted, list)
            and len(gold) == len(predicted)
            and all(_structured_equal(left, right) for left, right in zip(gold, predicted))
        )
    if isinstance(gold, dict):
        return (
            isinstance(predicted, dict)
            and gold.keys() == predicted.keys()
            and all(_structured_equal(gold[key], predicted[key]) for key in gold)
        )
    return False
