from datetime import date
from decimal import Decimal

import pytest

from datamate.normalization import (
    ColumnCoercion,
    ValueNormalizationError,
    normalize_rows,
    normalize_value,
)


def test_number_coercion_uses_declared_locale_and_exact_decimal() -> None:
    policy = ColumnCoercion(
        "number",
        decimal_separator=",",
        thousands_separator=".",
    )

    assert normalize_value("1.234,50", policy, column="amount") == Decimal("1234.50")
    assert normalize_value("2,50", policy, column="amount") == Decimal("2.50")
    assert normalize_value("", policy, column="amount") is None


def test_identifier_and_date_coercions_preserve_explicit_semantics() -> None:
    identifier = ColumnCoercion("identifier")
    day_first = ColumnCoercion("date", date_format="%d/%m/%Y")

    assert normalize_value("000012", identifier, column="code") == "000012"
    assert normalize_value("31/01/2026", day_first, column="date") == date(2026, 1, 31)


def test_normalization_fails_closed_for_bad_grouping_and_ambiguous_dates() -> None:
    number = ColumnCoercion("number", decimal_separator=",", thousands_separator=".")
    iso_date = ColumnCoercion("date")

    with pytest.raises(ValueNormalizationError, match="invalid thousands grouping"):
        normalize_value("12.34,50", number, column="amount")
    with pytest.raises(ValueNormalizationError, match="declared format"):
        normalize_value("31/01/2026", iso_date, column="date")


def test_rows_are_normalized_lazily_and_width_errors_are_explicit() -> None:
    rows = normalize_rows(
        [("12.50", "yes"), ("", "no")],
        ("amount", "active"),
        {
            "amount": ColumnCoercion("number"),
            "active": ColumnCoercion("boolean"),
        },
    )

    assert next(iter(rows)) == (Decimal("12.50"), True)
    with pytest.raises(ValueNormalizationError, match="Row width mismatch"):
        list(normalize_rows([("one",)], ("a", "b"), {}))


def test_normalization_rejects_duplicate_columns_and_unknown_policy_names() -> None:
    with pytest.raises(ValueNormalizationError, match="Duplicate column names"):
        list(normalize_rows([], ("amount", "amount"), {}))
    with pytest.raises(ValueNormalizationError, match="unknown columns"):
        list(
            normalize_rows(
                [],
                ("amount",),
                {"missing": ColumnCoercion("number")},
            )
        )
