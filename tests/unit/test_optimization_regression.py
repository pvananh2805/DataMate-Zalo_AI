import time
from pathlib import Path
from unittest.mock import MagicMock
import torch

from datamate.data_loader import BusinessDataLoader, _DISCOVERY_CACHE, _ROW_CACHE
from datamate.local_inference import JsonClosingStoppingCriteria
from datamate.question_understanding import _prune_schema_for_prompt


def test_discovery_cache_and_invalidation(tmp_path: Path):
    biz = tmp_path / "biz"
    biz.mkdir()
    f1 = biz / "data1.csv"
    f1.write_text("a,b\n1,2\n", encoding="utf-8")

    loader = BusinessDataLoader(biz)
    tables1 = loader.discover()
    assert len(tables1) == 1
    assert tables1[0].name == "data1"

    # Second call should hit discovery cache
    tables2 = loader.discover()
    assert tables1 is tables2

    # Add a new file -> cache must invalidate
    time.sleep(0.01)
    f2 = biz / "data2.csv"
    f2.write_text("c,d\n3,4\n", encoding="utf-8")

    tables3 = loader.discover()
    assert len(tables3) == 2
    assert {t.name for t in tables3} == {"data1", "data2"}


def test_row_cache_and_invalidation(tmp_path: Path):
    biz = tmp_path / "biz"
    biz.mkdir()
    f = biz / "sales.csv"
    f.write_text("item,qty\nA,10\nB,20\n", encoding="utf-8")

    loader = BusinessDataLoader(biz)
    (table,) = loader.discover()

    rows1 = list(loader.iter_rows(table))
    assert rows1 == [("A", "10"), ("B", "20")]

    # Second read hits row cache
    rows2 = list(loader.iter_rows(table))
    assert rows2 == rows1

    # Modify file -> row cache must invalidate
    time.sleep(0.01)
    f.write_text("item,qty\nA,10\nB,20\nC,30\n", encoding="utf-8")

    rows3 = list(loader.iter_rows(table))
    assert rows3 == [("A", "10"), ("B", "20"), ("C", "30")]


def test_early_stopping_criteria_handles_nested_json_and_strings():
    tokenizer = MagicMock()
    # Mock decoder that returns whatever text we simulate
    stopping = JsonClosingStoppingCriteria(tokenizer, p_len=5)

    # 1. Unbalanced JSON: {"a": {"b": 1}
    tokenizer.decode.side_effect = lambda tokens, **kw: '{"a": {"b": 1}' if len(tokens) > 1 else '}'
    input_ids = torch.zeros((1, 15), dtype=torch.long)
    assert not stopping(input_ids, None)

    # 2. Balanced JSON: {"a": {"b": 1}}
    tokenizer.decode.side_effect = lambda tokens, **kw: '{"a": {"b": 1}}' if len(tokens) > 1 else '}'
    assert stopping(input_ids, None)

    # 3. Brace inside string: {"text": "hello } world"}
    tokenizer.decode.side_effect = lambda tokens, **kw: '{"text": "hello } world"}' if len(tokens) > 1 else '}'
    assert stopping(input_ids, None)

    # 4. Incomplete brace inside string: {"text": "hello } world"
    tokenizer.decode.side_effect = lambda tokens, **kw: '{"text": "hello } world"' if len(tokens) > 1 else '}'
    assert not stopping(input_ids, None)


def test_prompt_pruning_filters_monthly_tables_and_preserves_dimensions():
    schema = {
        "BanHang_T1_2025.xlsx::Sheet1": ("Ngày", "Tiền"),
        "BanHang_T2_2025.xlsx::Sheet1": ("Ngày", "Tiền"),
        "BanHang_T11_2025.xlsx::Sheet1": ("Ngày", "Tiền"),
        "BanHang_T12_2025.xlsx::Sheet1": ("Ngày", "Tiền"),
        "DanhSachQuyDoi.xlsx::Sheet1": ("Mã", "Tên"),
    }
    pruned = _prune_schema_for_prompt("Tháng 11/2025 bán được bao nhiêu?", "2025-11-20", schema)
    # Should keep T11_2025 and DanhSachQuyDoi
    assert "BanHang_T11_2025.xlsx::Sheet1" in pruned
    assert "DanhSachQuyDoi.xlsx::Sheet1" in pruned
    assert "BanHang_T1_2025.xlsx::Sheet1" not in pruned
