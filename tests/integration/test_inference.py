import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from datamate.data_loader import BusinessDataLoader
from datamate.normalization import ColumnCoercion
from datamate.query_plan import Aggregation, ColumnRef
from datamate.question_understanding import JsonPlanAdapter
from datamate.run_logging import read_run_records
from datamate.submission import SubmissionWriter
from scripts.run_inference import (
    _process_questions,
    _read_coercions,
    _read_inference_config,
)


def test_inference_config_resolves_relative_paths_from_config_file(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "settings" / "inference.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "input": "questions.jsonl",
                "business_data_root": "business-data",
                "model_path": "models/checkpoint",
                "log": "output/run.jsonl",
                "submission_output": "output/submission.jsonl",
                "as_of_date": "2026-01-31",
                "precision": "fp16",
            }
        ),
        encoding="utf-8",
    )

    config = _read_inference_config(config_path)

    assert config["input"] == config_path.parent / "questions.jsonl"
    assert config["business_data_root"] == config_path.parent / "business-data"
    assert config["model_path"] == config_path.parent / "models" / "checkpoint"
    assert config["submission_output"] == (
        config_path.parent / "output" / "submission.jsonl"
    )
    assert config["as_of_date"] == "2026-01-31"
    assert config["precision"] == "fp16"


def test_inference_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "inference.json"
    config_path.write_text('{"unexpected": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown inference configuration fields"):
        _read_inference_config(config_path)


def test_coercion_config_reads_explicit_sparse_group_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "coercions.json"
    config_path.write_text(
        json.dumps(
            {
                "biz_1": {
                    "orders.csv::orders": {
                        "order_id": {
                            "kind": "identifier",
                            "fill_down_group": "order_id",
                            "deduplicate_group": "order_id",
                        },
                        "status": {
                            "kind": "text",
                            "fill_down_group": "order_id",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    policies = _read_coercions(config_path)

    assert policies == {
        "biz_1": {
            ColumnRef("orders.csv::orders", "order_id"): ColumnCoercion(
                "identifier",
                fill_down_group="order_id",
                deduplicate_group="order_id",
            ),
            ColumnRef("orders.csv::orders", "status"): ColumnCoercion(
                "text",
                fill_down_group="order_id",
            ),
        }
    }


def test_inference_processor_integrates_context_logging_and_submission(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    business_root = dataset_root / "data"
    business = business_root / "biz_1"
    business.mkdir(parents=True)
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute("CREATE TABLE sales (amount INTEGER)")
        connection.executemany("INSERT INTO sales VALUES (?)", [(12,), (8,)])
    rulebook = dataset_root / "rulebook.md"
    rulebook.write_text("# Quy tắc dùng trong kiểm thử\n", encoding="utf-8")
    (business / "README.md").write_text(
        "amount được tính bằng VND.\n",
        encoding="utf-8",
    )
    with (dataset_root / "calendar.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["solar_date"])
        writer.writerow(["2026-01-31"])

    table = next(
        item
        for item in BusinessDataLoader(business).discover()
        if item.name == "sales"
    )
    table_id = f"{table.relative_path}::{table.sqlite_table}"
    plan = {
        "base_table": table_id,
        "aggregations": [
            {
                "function": "sum",
                "column": {"table": table_id, "column": "amount"},
                "alias": "revenue",
            }
        ],
        "output_type": "number",
    }
    prompts: list[str] = []
    planner = JsonPlanAdapter(
        lambda prompt: prompts.append(prompt) or json.dumps(plan),
        model_name="test-backend",
    )
    questions = tmp_path / "questions" / "questions.jsonl"
    questions.parent.mkdir()
    questions.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in (
                {
                    "id": "q-default-date",
                    "business_id": "biz_1",
                    "question": "Tổng doanh thu?",
                },
                {
                    "id": "q-explicit-date",
                    "business_id": "biz_1",
                    "question": "Tổng doanh thu?",
                    "as_of_date": "2026-02-01",
                },
            )
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "logs" / "runs.jsonl"
    submission_path = tmp_path / "result" / "submission.jsonl"
    submission_writer = SubmissionWriter(submission_path)

    succeeded, failed = _process_questions(
        questions,
        business_root,
        planner,
        log_path,
        {},
        rulebook_path=None,
        calendar_path=None,
        submission_writer=submission_writer,
        default_as_of_date="2026-01-31",
    )
    assert (succeeded, failed) == (2, 0)
    submission_writer.commit()

    assert "# Quy tắc dùng trong kiểm thử" in prompts[0]
    assert "amount được tính bằng VND." in prompts[0]
    assert "2026-01-31" in prompts[0]
    assert "2026-02-01" in prompts[1]
    run_records = read_run_records(log_path)
    assert len(run_records) == 2
    run_record = run_records[0]
    assert (
        run_record["input"]["metadata"]["business_context_sources"]["rulebook"][
            "sha256"
        ]
        == hashlib.sha256(rulebook.read_bytes()).hexdigest()
    )
    assert [
        json.loads(line)
        for line in submission_path.read_text(encoding="utf-8").splitlines()
    ] == [
        {"id": "q-default-date", "type": "number", "answer": 20},
        {"id": "q-explicit-date", "type": "number", "answer": 20},
    ]
