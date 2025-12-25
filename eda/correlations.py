"""
Analyzes correlations between columns.
"""

from __future__ import annotations

import pandas as pd
from typing import Literal


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of numeric column names."""

    return df.select_dtypes(include="number").columns.tolist()


def analyze_correlations(
    df: pd.DataFrame,
    *,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    max_features: int = 30,
) -> pd.DataFrame:
    """
    Calculates the correlation matrix for numerical columns.

    Args:
        df: The input DataFrame.

    Returns:
        A DataFrame with the correlation matrix.
    """
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        return pd.DataFrame()

    if max_features is not None and max_features > 0:
        numeric_cols = numeric_cols[: int(max_features)]

    numeric_df = df[numeric_cols]

    corr = numeric_df.corr(method=method)
    return corr
