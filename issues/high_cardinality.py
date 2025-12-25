"""
Detects columns with high cardinality.
"""

from __future__ import annotations

import pandas as pd


def detect_high_cardinality(
    df: pd.DataFrame,
    *,
    column_types: dict[str, str] | None = None,
    target_column: str | None = None,
    threshold: int = 50,
    max_columns: int = 50,
) -> dict:
    """
    Detects categorical columns with high cardinality.

    Args:
        df: The input DataFrame.
        threshold: The cardinality threshold.

    Returns:
        A dictionary with a detailed high-cardinality report.
    """
    excluded = {str(target_column)} if target_column else set()

    if column_types:
        categorical_cols = [
            c for c, t in column_types.items() if t == "categorical" and c in df.columns and c not in excluded
        ]
    else:
        categorical_cols = df.select_dtypes(
            include=["object", "category"]).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in excluded]
    warnings: list[str] = []
    if max_columns is not None and max_columns > 0 and len(categorical_cols) > max_columns:
        warnings.append(
            f"Too many categorical columns ({len(categorical_cols)}). Scanning first {max_columns}."
        )
        categorical_cols = categorical_cols[: int(max_columns)]

    rows: list[dict] = []
    total_rows = len(df) or 1
    for col in categorical_cols:
        series = df[col]
        unique_values = int(series.nunique(dropna=True))
        unique_percent = (unique_values / total_rows) * 100.0
        if unique_values > int(threshold):
            value_counts = series.value_counts(dropna=False)
            top_value = value_counts.idxmax()
            top_count = int(value_counts.iloc[0])
            top_percent = (top_count / total_rows) * 100.0
            bottom_value = value_counts.idxmin()
            bottom_count = int(value_counts.iloc[-1])
            bottom_percent = (bottom_count / total_rows) * 100.0
            sample = series.dropna().astype(str).unique()[:5].tolist()
            rows.append(
                {
                    "column": col,
                    "unique_values": unique_values,
                    "unique_percent": float(unique_percent),
                    "top_value": str(top_value),
                    "top_count": top_count,
                    "top_percent": float(top_percent),
                    "bottom_value": str(bottom_value),
                    "bottom_count": bottom_count,
                    "bottom_percent": float(bottom_percent),
                    "sample_values": ", ".join(sample),
                }
            )

    table = (
        pd.DataFrame(rows)
        .sort_values(by=["unique_values", "unique_percent"], ascending=False)
        .reset_index(drop=True)
    ) if rows else pd.DataFrame(
        columns=[
            "column",
            "unique_values",
            "unique_percent",
            "top_value",
            "top_count",
            "top_percent",
            "bottom_value",
            "bottom_count",
            "bottom_percent",
            "sample_values",
        ]
    )

    return {
        "threshold": int(threshold),
        "warnings": warnings,
        "scanned_columns": categorical_cols,
        "table": table,
        "flagged_columns": table["column"].tolist() if not table.empty else [],
    }
