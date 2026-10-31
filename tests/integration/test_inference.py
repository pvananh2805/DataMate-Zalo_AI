import csv
import hashlib
import json
import sqlite3
from pathlib import Path

from datamate.data_loader import BusinessDataLoader
from datamate.query_plan import Aggregation, ColumnRef
from datamate.question_understanding import JsonPlanAdapter
from datamate.run_logging import read_run_records
from datamate.submission import SubmissionWriter
from scripts.run_inference import _process_questions


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
        json.dumps(
            {
                "id": "q-1",
                "business_id": "biz_1",
                "question": "Tổng doanh thu?",
                "as_of_date": "2026-01-31",
            },
            ensure_ascii=False,
        )
        + "\n",
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
    )
    assert (succeeded, failed) == (1, 0)
    submission_writer.commit()

    assert "# Quy tắc dùng trong kiểm thử" in prompts[0]
    assert "amount được tính bằng VND." in prompts[0]
    assert "2026-01-31" in prompts[0]
    (run_record,) = read_run_records(log_path)
    assert (
        run_record["input"]["metadata"]["business_context_sources"]["rulebook"][
            "sha256"
        ]
        == hashlib.sha256(rulebook.read_bytes()).hexdigest()
    )
    assert json.loads(submission_path.read_text(encoding="utf-8")) == {
        "id": "q-1",
        "type": "number",
        "answer": 20,
    }
