"""
Extracts and displays metadata from the dataset.
"""

from __future__ import annotations

import pandas as pd


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Gets basic information about the dataset.

    Args:
        df: The input DataFrame.

    Returns:
        A dictionary with dataset information.
    """
    return {
        "shape": get_shape(df),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict()
    }


def get_shape(df: pd.DataFrame) -> tuple[int, int]:
    """Return the dataset shape as (rows, cols)."""

    rows, cols = df.shape
    return int(rows), int(cols)


def get_preview(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the first `n` rows of the dataset (non-mutating)."""

    if n <= 0:
        return df.head(0)
    return df.head(int(n))


def get_target_class_distribution(
    df: pd.DataFrame,
    target_column: str,
    *,
    normalize: bool = False,
    dropna: bool = False,
) -> pd.DataFrame:
    """Compute target column class distribution.

    Args:
        df: Input dataset.
        target_column: Name of the target column.
        normalize: If True, returns proportions instead of counts.
        dropna: If True, ignores missing targets.

    Returns:
        DataFrame with columns: class, count (or proportion).
    """

    if target_column not in df.columns:
        raise KeyError(
            f"Target column {target_column!r} not found in dataset.")

    series = df[target_column]
    counts = series.value_counts(dropna=dropna, normalize=normalize)

    value_col = "proportion" if normalize else "count"
    return (
        counts.rename(value_col)
        .reset_index()
        .rename(columns={"index": "class"})
    )
