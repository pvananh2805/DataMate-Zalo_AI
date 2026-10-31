import json
from pathlib import Path

from datamate.evaluation import evaluate_predictions


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_local_evaluation_reports_types_missing_extra_and_unanswerable(
    tmp_path: Path,
) -> None:
    gold_path = tmp_path / "gold" / "answers.jsonl"
    prediction_path = tmp_path / "predictions" / "runs.jsonl"
    questions_path = tmp_path / "questions" / "questions.jsonl"
    gold_path.parent.mkdir()
    prediction_path.parent.mkdir()
    questions_path.parent.mkdir()
    _write_jsonl(
        gold_path,
        [
            {"id": "q1", "type": "number", "answer": 10},
            {"id": "q2", "type": "unanswerable", "answer": None},
            {"id": "q3", "type": "percent", "answer": 2.08},
        ],
    )
    _write_jsonl(
        prediction_path,
        [
            {
                "question_id": "q1",
                "business_id": "biz_1",
                "answer": {
                    "answer_type": "number",
                    "value": "10.00",
                    "status": "answered",
                },
                "elapsed_ms": 12,
            },
            {
                "question_id": "q2",
                "business_id": "biz_2",
                "answer": {
                    "answer_type": "unanswerable",
                    "value": None,
                    "status": "no_data",
                },
                "elapsed_ms": 8,
            },
            {"id": "extra", "type": "number", "answer": 99},
        ],
    )
    _write_jsonl(
        questions_path,
        [
            {"id": "q1", "business_id": "biz_1"},
            {"id": "q2", "business_id": "biz_2"},
            {"id": "q3", "business_id": "biz_2"},
        ],
    )

    report = evaluate_predictions(
        gold_path,
        prediction_path,
        tmp_path / "reports" / "evaluation.json",
        questions_path=questions_path,
    )

    assert report["metric"] == "local_strict_exact_match_not_official_PA"
    assert report["official_pa_computed"] is False
    assert report["exact_matches"] == 2
    assert report["exact_match_fraction"] == 2 / 3
    assert report["missing_predictions"] == 1
    assert report["extra_predictions"] == 1
    assert report["unanswerable_exact_matches"] == 1
    assert report["latency_ms"]["average"] == 10
    assert report["by_business"]["biz_2"]["count"] == 2


def test_local_evaluation_compares_numeric_items_in_logged_lists_exactly(
    tmp_path: Path,
) -> None:
    (tmp_path / "gold").mkdir()
    (tmp_path / "predictions").mkdir()
    gold_path = tmp_path / "gold" / "answers.jsonl"
    prediction_path = tmp_path / "predictions" / "runs.jsonl"
    _write_jsonl(
        gold_path,
        [{"id": "q1", "type": "list", "answer": [10, 20.5]}],
    )
    _write_jsonl(
        prediction_path,
        [
            {
                "id": "q1",
                "type": "list",
                "answer": ["10.00", "20.500"],
            }
        ],
    )

    report = evaluate_predictions(
        gold_path,
        prediction_path,
        tmp_path / "reports" / "report.json",
    )

    assert report["exact_matches"] == 1
