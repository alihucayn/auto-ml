"""Detects data type mismatches in columns, respecting declared column types."""

from __future__ import annotations

import pandas as pd


def detect_datatype_mismatch(
    df: pd.DataFrame,
    *,
    column_types: dict[str, str] | None = None,
    target_column: str | None = None,
    convertible_fraction_threshold: float = 0.90,
    sample_size: int = 5000,
) -> dict:
    """
    Detect potential data type mismatches.

    - If a column is marked numeric but stored as strings, flag it.
    - If a column is marked categorical but looks numeric, flag it.
    - For untyped columns, flag numeric- or datetime-like strings.
    """

    excluded = {str(target_column)} if target_column else set()
    rows: list[dict] = []

    for col in df.columns:
        if col in excluded:
            continue

        series = df[col]
        declared = (column_types or {}).get(col)

        non_null = series.dropna()
        if non_null.empty:
            continue

        if sample_size is not None and sample_size > 0 and len(non_null) > sample_size:
            non_null = non_null.sample(n=int(sample_size), random_state=42)

        text = non_null.astype(str).str.strip()
        if text.empty:
            continue

        numeric_conv = pd.to_numeric(text, errors="coerce")
        numeric_fraction = float(numeric_conv.notna().mean())

        datetime_conv = pd.to_datetime(text, errors="coerce")
        datetime_fraction = float(datetime_conv.notna().mean())

        is_numeric_dtype = pd.api.types.is_numeric_dtype(series)
        is_datetime_dtype = pd.api.types.is_datetime64_any_dtype(series)

        issue_type: str | None = None
        detail: str | None = None
        confidence: float = 0.0

        if declared == "numeric":
            if not is_numeric_dtype and numeric_fraction >= float(convertible_fraction_threshold):
                issue_type = "declared_numeric_string_values"
                detail = "Column marked numeric contains string values convertible to numbers."
                confidence = numeric_fraction
            elif not is_numeric_dtype:
                issue_type = "declared_numeric_mixed_types"
                detail = "Column marked numeric contains non-numeric values that are not convertible."
                confidence = 1.0 - numeric_fraction
        elif declared == "categorical":
            if is_numeric_dtype and numeric_fraction >= float(convertible_fraction_threshold):
                issue_type = "categorical_as_numeric_dtype"
                detail = "Column marked categorical is stored as numeric; ensure encoding is intentional."
                confidence = numeric_fraction
            elif numeric_fraction >= float(convertible_fraction_threshold):
                issue_type = "categorical_looks_numeric"
                detail = "Categorical column values are mostly numeric-like strings; consider retyping."
                confidence = numeric_fraction
        else:
            if numeric_fraction >= float(convertible_fraction_threshold) and not is_numeric_dtype:
                issue_type = "numeric_like_strings"
                detail = "Untyped column contains mostly numeric-like strings; consider setting to numeric."
                confidence = numeric_fraction
            elif datetime_fraction >= float(convertible_fraction_threshold) and not is_datetime_dtype:
                issue_type = "datetime_like_strings"
                detail = "Untyped column contains mostly datetime-like strings; consider parsing to datetime."
                confidence = datetime_fraction

        if issue_type is not None:
            rows.append(
                {
                    "column": col,
                    "declared_type": declared or "unknown",
                    "issue_type": issue_type,
                    "numeric_convertible_fraction": numeric_fraction,
                    "datetime_convertible_fraction": datetime_fraction,
                    "detail": detail or "",
                    "confidence": float(confidence),
                }
            )

    table = (
        pd.DataFrame(rows)
        .sort_values(by=["confidence", "column"], ascending=[False, True])
        .reset_index(drop=True)
    ) if rows else pd.DataFrame(
        columns=[
            "column",
            "declared_type",
            "issue_type",
            "numeric_convertible_fraction",
            "datetime_convertible_fraction",
            "detail",
            "confidence",
        ]
    )

    return {
        "threshold": float(convertible_fraction_threshold),
        "sample_size": int(sample_size),
        "table": table,
        "flagged_columns": table["column"].tolist() if not table.empty else [],
    }
