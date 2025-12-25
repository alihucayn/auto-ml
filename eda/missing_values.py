"""
Analyzes missing values in the dataset.
"""

from __future__ import annotations

import pandas as pd


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a per-column missing values summary.

    Args:
        df: Input dataset.

    Returns:
        DataFrame with columns: column, missing_count, missing_percent.
    """

    total_rows = len(df)
    missing_count = df.isna().sum()
    if total_rows == 0:
        missing_percent = missing_count * 0.0
    else:
        missing_percent = (missing_count / total_rows) * 100.0

    table = (
        pd.DataFrame(
            {
                "missing_count": missing_count,
                "missing_percent": missing_percent,
            }
        )
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values(by=["missing_percent", "missing_count"], ascending=False)
        .reset_index(drop=True)
    )
    return table


def analyze_missing_values(df: pd.DataFrame) -> dict:
    """
    Analyzes missing values in the dataset.

    Args:
        df: The input DataFrame.

    Returns:
        A dictionary with missing value analysis.
    """
    table = missing_values_table(df)
    total_missing = int(df.isna().sum().sum())
    return {
        "table": table,
        "total_missing": total_missing,
    }
