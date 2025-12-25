"""
Detects outliers in the dataset.
"""

from __future__ import annotations

import pandas as pd


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _iqr_outlier_mask(series: pd.Series, *, multiplier: float) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)
    lower = q1 - (multiplier * iqr)
    upper = q3 + (multiplier * iqr)
    return (series < lower) | (series > upper)


def _zscore_outlier_mask(series: pd.Series, *, threshold: float) -> pd.Series:
    values = series.astype(float)
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(False, index=series.index)
    z = (values - mean) / std
    return z.abs() > threshold


def detect_outliers(
    df: pd.DataFrame,
    *,
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
    max_columns: int = 30,
) -> dict:
    """
    Detects outliers in numerical columns.

    Args:
        df: The input DataFrame.

    Returns:
        A dictionary with outlier information.
    """
    numeric_cols = _numeric_columns(df)
    warnings: list[str] = []
    if max_columns is not None and max_columns > 0 and len(numeric_cols) > max_columns:
        warnings.append(
            f"Too many numeric columns for outlier scan ({len(numeric_cols)}). "
            f"Scanning first {max_columns}."
        )
        numeric_cols = numeric_cols[: int(max_columns)]

    total_rows = len(df)
    rows: list[dict] = []

    method_lower = method.strip().lower()
    for col in numeric_cols:
        series = df[col]
        non_null = series.dropna()
        if non_null.empty:
            rows.append(
                {
                    "column": col,
                    "method": method_lower,
                    "outlier_count": 0,
                    "outlier_percent": 0.0,
                    "non_null_count": 0,
                }
            )
            continue

        if method_lower == "iqr":
            mask = _iqr_outlier_mask(
                non_null, multiplier=float(iqr_multiplier))
        elif method_lower in {"z", "zscore", "z-score"}:
            mask = _zscore_outlier_mask(
                non_null, threshold=float(zscore_threshold))
        else:
            raise ValueError("Outlier method must be 'iqr' or 'zscore'.")

        outlier_count = int(mask.sum())
        denom = int(non_null.shape[0]) if non_null.shape[0] else (
            total_rows or 1)
        outlier_percent = (outlier_count / denom) * 100.0
        rows.append(
            {
                "column": col,
                "method": method_lower,
                "outlier_count": outlier_count,
                "outlier_percent": float(outlier_percent),
                "non_null_count": int(non_null.shape[0]),
            }
        )

    table = (
        pd.DataFrame(rows)
        .sort_values(by=["outlier_count", "outlier_percent"], ascending=False)
        .reset_index(drop=True)
    )
    return {
        "warnings": warnings,
        "table": table,
        "scanned_columns": numeric_cols,
    }
