"""
Analyzes the distribution of data in columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def analyze_distributions(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    bins: int = 30,
    max_columns: int = 20,
    max_categories: int = 50,
) -> dict:
    """
    Analyzes the distribution of data in each column.

    Args:
        df: The input DataFrame.

    Returns:
        A dictionary with distribution plots or statistics.
    """
    if columns is None:
        columns = _numeric_columns(df)

    cat_cols = [] if categorical_columns is None else [
        c for c in categorical_columns if c in df.columns]

    warnings: list[str] = []
    if max_columns is not None and max_columns > 0 and len(columns) > max_columns:
        warnings.append(
            f"Too many columns selected for numeric distributions ({len(columns)}). "
            f"Showing first {max_columns}."
        )
        columns = columns[: int(max_columns)]

    if max_columns is not None and max_columns > 0 and len(cat_cols) > max_columns:
        warnings.append(
            f"Too many categorical columns selected for distributions ({len(cat_cols)}). "
            f"Showing first {max_columns}."
        )
        cat_cols = cat_cols[: int(max_columns)]

    results: dict[str, dict] = {}

    # Numeric histograms
    for col in columns:
        if col not in df.columns:
            continue

        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        values = series.dropna().to_numpy()
        if values.size == 0:
            results[col] = {
                "n": 0,
                "bins": [],
                "counts": [],
                "stats": {"mean": None, "std": None, "min": None, "max": None},
            }
            continue

        counts, edges = np.histogram(values, bins=int(bins))
        stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        results[col] = {
            "n": int(values.size),
            "bins": edges.tolist(),
            "counts": counts.tolist(),
            "stats": stats,
        }

    # Categorical value counts
    for col in cat_cols:
        if col not in df.columns:
            continue
        series = df[col].astype(str)
        vc = series.value_counts(dropna=False).reset_index()
        vc.columns = ["value", "count"]
        vc = vc.head(max_categories)
        results[col] = {
            "n": int(len(series)),
            "value_counts": vc.values.tolist(),
            "stats": None,
        }

    return {
        "warnings": warnings,
        "columns": columns,
        "categorical_columns": cat_cols,
        "results": results,
    }
