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
    assert report["incorrect_predictions"] == 0
    assert report["abstentions"] == 1
    assert report["strict_local_reward_sum"] == 2
    assert report["official_pa_computed"] is False
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
                "answer": {
                    "answer_type": "list",
                    "value": ["10.00", "20.500"],
                    "status": "answered",
                },
            }
        ],
    )

    report = evaluate_predictions(
        gold_path,
        prediction_path,
        tmp_path / "reports" / "report.json",
    )

    assert report["exact_matches"] == 1


def test_submission_jsonl_rejects_numeric_strings_and_scores_wrong_answer(
    tmp_path: Path,
) -> None:
    (tmp_path / "gold").mkdir()
    (tmp_path / "predictions").mkdir()
    gold_path = tmp_path / "gold" / "answers.jsonl"
    prediction_path = tmp_path / "predictions" / "submission.jsonl"
    _write_jsonl(
        gold_path,
        [{"id": "q1", "type": "number", "answer": 10}],
    )
    _write_jsonl(
        prediction_path,
        [{"id": "q1", "type": "number", "answer": "10"}],
    )

    report = evaluate_predictions(
        gold_path,
        prediction_path,
        tmp_path / "reports" / "report.json",
    )

    assert report["exact_matches"] == 0
    assert report["invalid_predictions"] == 1
    assert report["incorrect_predictions"] == 1
    assert report["strict_local_reward_sum"] == -1


def test_explicit_abstention_receives_zero_rulebook_reward(
    tmp_path: Path,
) -> None:
    (tmp_path / "gold").mkdir()
    (tmp_path / "predictions").mkdir()
    gold_path = tmp_path / "gold" / "answers.jsonl"
    prediction_path = tmp_path / "predictions" / "submission.jsonl"
    _write_jsonl(
        gold_path,
        [{"id": "q1", "type": "number", "answer": 10}],
    )
    _write_jsonl(
        prediction_path,
        [{"id": "q1", "type": "abstain"}],
    )

    report = evaluate_predictions(
        gold_path,
        prediction_path,
        tmp_path / "reports" / "report.json",
    )

    assert report["abstentions"] == 1
    assert report["incorrect_predictions"] == 0
    assert report["strict_local_reward_sum"] == 0


def test_malformed_abstention_is_not_scored_as_valid_abstention(
    tmp_path: Path,
) -> None:
    (tmp_path / "gold").mkdir()
    (tmp_path / "predictions").mkdir()
    gold_path = tmp_path / "gold" / "answers.jsonl"
    prediction_path = tmp_path / "predictions" / "submission.jsonl"
    _write_jsonl(
        gold_path,
        [{"id": "q1", "type": "number", "answer": 10}],
    )
    _write_jsonl(
        prediction_path,
        [{"id": "q1", "type": "abstain", "answer": 10}],
    )

    report = evaluate_predictions(
        gold_path,
        prediction_path,
        tmp_path / "reports" / "report.json",
    )

    assert report["abstentions"] == 0
    assert report["incorrect_predictions"] == 1
    assert report["strict_local_reward_sum"] == -1


def test_submission_jsonl_accepts_rulebook_unanswerable_payload(
    tmp_path: Path,
) -> None:
    (tmp_path / "gold").mkdir()
    (tmp_path / "predictions").mkdir()
    gold_path = tmp_path / "gold" / "answers.jsonl"
    prediction_path = tmp_path / "predictions" / "submission.jsonl"
    _write_jsonl(
        gold_path,
        [{"id": "q1", "type": "unanswerable", "answer": None}],
    )
    _write_jsonl(
        prediction_path,
        [
            {
                "id": "q1",
                "type": "unanswerable",
                "answer": {
                    "type": "unanswerable",
                    "reason": "Dữ liệu không có thông tin này",
                },
            }
        ],
    )

    report = evaluate_predictions(
        gold_path,
        prediction_path,
        tmp_path / "reports" / "report.json",
    )

    assert report["exact_matches"] == 1
    assert report["invalid_predictions"] == 0


def test_submission_jsonl_accepts_year_month_date(
    tmp_path: Path,
) -> None:
    (tmp_path / "gold").mkdir()
    (tmp_path / "predictions").mkdir()
    gold_path = tmp_path / "gold" / "answers.jsonl"
    prediction_path = tmp_path / "predictions" / "submission.jsonl"
    _write_jsonl(
        gold_path,
        [{"id": "q1", "type": "date", "answer": "2025-09"}],
    )
    _write_jsonl(
        prediction_path,
        [{"id": "q1", "type": "date", "answer": "2025-09"}],
    )

    report = evaluate_predictions(
        gold_path,
        prediction_path,
        tmp_path / "reports" / "report.json",
    )

    assert report["exact_matches"] == 1


def test_submission_jsonl_requires_columns_and_rows_for_table(
    tmp_path: Path,
) -> None:
    (tmp_path / "gold").mkdir()
    (tmp_path / "predictions").mkdir()
    gold_path = tmp_path / "gold" / "answers.jsonl"
    prediction_path = tmp_path / "predictions" / "submission.jsonl"
    _write_jsonl(
        gold_path,
        [
            {
                "id": "q1",
                "type": "table",
                "answer": {"columns": ["name"], "rows": [["Trà"]]},
            }
        ],
    )
    _write_jsonl(
        prediction_path,
        [{"id": "q1", "type": "table", "answer": [["Trà"]]}],
    )

    report = evaluate_predictions(
        gold_path,
        prediction_path,
        tmp_path / "reports" / "report.json",
    )

    assert report["exact_matches"] == 0
    assert report["invalid_predictions"] == 1
    assert report["strict_local_reward_sum"] == -1


def test_string_alias_and_case_whitespace_punctuation_are_accepted(
    tmp_path: Path,
) -> None:
    (tmp_path / "gold").mkdir()
    (tmp_path / "predictions").mkdir()
    gold_path = tmp_path / "gold" / "answers.jsonl"
    prediction_path = tmp_path / "predictions" / "submission.jsonl"
    _write_jsonl(
        gold_path,
        [
            {
                "id": "q1",
                "type": "string",
                "answer": "Sữa",
                "alias": ["Sữa tươi"],
            }
        ],
    )
    _write_jsonl(
        prediction_path,
        [{"id": "q1", "type": "string", "answer": "  SỮA-TƯƠI! "}],
    )

    report = evaluate_predictions(
        gold_path,
        prediction_path,
        tmp_path / "reports" / "report.json",
    )

    assert report["exact_matches"] == 1
    assert report["aliases_compared_for_strings"] is True
    assert report["string_alias_count"] == 1


def test_string_normalization_does_not_remove_vietnamese_diacritics(
    tmp_path: Path,
) -> None:
    (tmp_path / "gold").mkdir()
    (tmp_path / "predictions").mkdir()
    gold_path = tmp_path / "gold" / "answers.jsonl"
    prediction_path = tmp_path / "predictions" / "submission.jsonl"
    _write_jsonl(
        gold_path,
        [{"id": "q1", "type": "string", "answer": "Cà phê"}],
    )
    _write_jsonl(
        prediction_path,
        [{"id": "q1", "type": "string", "answer": "Ca phe"}],
    )

    report = evaluate_predictions(
        gold_path,
        prediction_path,
        tmp_path / "reports" / "report.json",
    )

    assert report["exact_matches"] == 0
    assert report["incorrect_predictions"] == 1


def test_table_evaluation_ignores_row_order_and_matches_columns_by_name(
    tmp_path: Path,
) -> None:
    (tmp_path / "gold").mkdir()
    (tmp_path / "predictions").mkdir()
    gold_path = tmp_path / "gold" / "answers.jsonl"
    prediction_path = tmp_path / "predictions" / "submission.jsonl"
    _write_jsonl(
        gold_path,
        [
            {
                "id": "q1",
                "type": "table",
                "answer": {
                    "columns": ["name", "revenue"],
                    "rows": [["Trà", 10], ["Cà phê", 20]],
                },
            }
        ],
    )
    _write_jsonl(
        prediction_path,
        [
            {
                "id": "q1",
                "type": "table",
                "answer": {
                    "columns": ["revenue", "name"],
                    "rows": [[20, "Cà phê"], [10, "Trà"]],
                },
            }
        ],
    )

    report = evaluate_predictions(
        gold_path,
        prediction_path,
        tmp_path / "reports" / "report.json",
    )

    assert report["exact_matches"] == 1


def test_table_evaluation_normalizes_internal_run_log_rows(
    tmp_path: Path,
) -> None:
    (tmp_path / "gold").mkdir()
    (tmp_path / "predictions").mkdir()
    gold_path = tmp_path / "gold" / "answers.jsonl"
    prediction_path = tmp_path / "predictions" / "runs.jsonl"
    _write_jsonl(
        gold_path,
        [
            {
                "id": "q1",
                "type": "table",
                "answer": {
                    "columns": ["name", "revenue"],
                    "rows": [["Trà", 10]],
                },
            }
        ],
    )
    _write_jsonl(
        prediction_path,
        [
            {
                "question_id": "q1",
                "query_plan": {
                    "base_table": "sales",
                    "group_by": [
                        {"table": "sales", "column": "name"},
                    ],
                    "aggregations": [
                        {
                            "function": "sum",
                            "column": {"table": "sales", "column": "amount"},
                            "alias": "revenue",
                        }
                    ],
                },
                "answer": {
                    "answer_type": "table",
                    "value": [{"sales.name": "Trà", "revenue": "10.00"}],
                    "status": "answered",
                },
            }
        ],
    )

    report = evaluate_predictions(
        gold_path,
        prediction_path,
        tmp_path / "reports" / "report.json",
    )

    assert report["exact_matches"] == 1
