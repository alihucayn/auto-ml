"""
Detects class imbalance in the target variable.
"""

from __future__ import annotations

import pandas as pd


def detect_imbalance(
    df: pd.DataFrame,
    target_column: str,
    *,
    min_class_proportion_threshold: float = 0.20,
    imbalance_ratio_threshold: float = 0.50,
    dropna: bool = True,
) -> dict:
    """
    Detects class imbalance in the target variable.

    Args:
        df: The input DataFrame.
        target_column: The name of the target column.

    Returns:
        A dictionary with imbalance information.
    """
    if target_column not in df.columns:
        raise KeyError(
            f"Target column {target_column!r} not found in dataset.")

                                                              
                                                                       
    target = df[target_column]
    missing_target_count = int(target.isna().sum())
    target_non_null = target.dropna()

                                                                                 
    _ = dropna

    counts = target_non_null.value_counts(dropna=True)
    proportions = target_non_null.value_counts(dropna=True, normalize=True)

    if counts.empty:
        return {
            "is_imbalanced": False,
            "reason": "Target column has no non-null values.",
            "missing_target_count": missing_target_count,
            "distribution": pd.DataFrame(columns=["class", "count", "proportion"]),
        }

    max_prop = float(proportions.max())
    min_prop = float(proportions.min())
    imbalance_ratio = (min_prop / max_prop) if max_prop > 0 else 1.0

    is_imbalanced = (
        min_prop < float(min_class_proportion_threshold)
        or imbalance_ratio < float(imbalance_ratio_threshold)
    )

    distribution = (
        pd.DataFrame({"count": counts, "proportion": proportions})
        .reset_index()
        .sort_values(by="count", ascending=False)
        .reset_index(drop=True)
    )

                                                                           
                                                         
    if not distribution.empty:
        first_col = distribution.columns[0]
        if first_col != "class":
            distribution = distribution.rename(columns={first_col: "class"})

    return {
        "is_imbalanced": bool(is_imbalanced),
        "missing_target_count": missing_target_count,
        "min_class_proportion": float(min_prop),
        "max_class_proportion": float(max_prop),
        "imbalance_ratio": float(imbalance_ratio),
        "thresholds": {
            "min_class_proportion_threshold": float(min_class_proportion_threshold),
            "imbalance_ratio_threshold": float(imbalance_ratio_threshold),
        },
        "distribution": distribution,
    }
