import json
import sqlite3
from pathlib import Path

import pytest

from datamate.training_data import (
    TrainingDataError,
    prepare_plan_training_data,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_business(root: Path, business_id: str) -> None:
    business = root / business_id
    business.mkdir(parents=True)
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute("CREATE TABLE sales (amount NUMERIC)")
        connection.execute("INSERT INTO sales VALUES (10)")


def _plan() -> dict[str, object]:
    return {
        "base_table": "shop.sqlite::sales",
        "aggregations": [
            {
                "function": "sum",
                "column": {
                    "table": "shop.sqlite::sales",
                    "column": "amount",
                },
                "alias": "total",
            }
        ],
        "output_type": "number",
    }


def test_training_data_uses_manual_plans_and_business_level_holdout(
    tmp_path: Path,
) -> None:
    questions_path = tmp_path / "source" / "questions.jsonl"
    annotations_path = tmp_path / "source" / "annotations.jsonl"
    questions_path.parent.mkdir()
    data_root = tmp_path / "businesses"
    data_root.mkdir()
    _make_business(data_root, "biz_1")
    _make_business(data_root, "biz_2")
    _write_jsonl(
        questions_path,
        [
            {
                "id": "q1",
                "business_id": "biz_1",
                "question": "Tổng doanh thu?",
                "as_of_date": "2026-01-31",
            },
            {
                "id": "q2",
                "business_id": "biz_2",
                "question": "Tổng tiền bán hàng?",
                "as_of_date": "2026-01-31",
            },
            {
                "id": "q3",
                "business_id": "biz_1",
                "question": "Câu chưa gán nhãn",
            },
        ],
    )
    _write_jsonl(
        annotations_path,
        [
            {"question_id": "q1", "business_id": "biz_1", "query_plan": _plan()},
            {"question_id": "q2", "business_id": "biz_2", "query_plan": _plan()},
        ],
    )
    output = tmp_path / "prepared"

    report = prepare_plan_training_data(
        questions_path,
        annotations_path,
        data_root,
        output,
        validation_fraction=0.5,
        seed=17,
    )

    assert report.question_count == 3
    assert report.annotation_count == 2
    assert report.unannotated_question_count == 1
    assert report.train_count == report.validation_count == 1
    assert set(report.train_business_ids).isdisjoint(report.validation_business_ids)
    examples = [
        json.loads(line)
        for split in ("train.jsonl", "validation.jsonl")
        for line in (output / split).read_text(encoding="utf-8").splitlines()
    ]
    assert {example["question_id"] for example in examples} == {"q1", "q2"}
    assert all("amount" in example["prompt"] for example in examples)
    assert (output / "preparation_report.json").is_file()


def test_training_data_rejects_mismatched_labels_and_preserves_no_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    questions_path = source / "questions.jsonl"
    annotations_path = source / "annotations.jsonl"
    _write_jsonl(
        questions_path,
        [
            {
                "id": "q1",
                "business_id": "biz_1",
                "question": "Tổng?",
            }
        ],
    )
    _write_jsonl(
        annotations_path,
        [{"question_id": "q1", "business_id": "biz_wrong", "query_plan": _plan()}],
    )
    data_root = tmp_path / "businesses"
    data_root.mkdir()
    output = tmp_path / "prepared"

    with pytest.raises(TrainingDataError, match="business_id does not match"):
        prepare_plan_training_data(
            questions_path,
            annotations_path,
            data_root,
            output,
        )

    assert not output.exists()
