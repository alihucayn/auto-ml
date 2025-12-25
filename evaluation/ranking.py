"""
Ranks models based on performance.
"""

from __future__ import annotations

import pandas as pd


def rank_models(comparison_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Ranks models based on a specified metric.

    Args:
        comparison_df: The DataFrame with model comparison results.
        metric: The metric to rank by.

    Returns:
        A ranked DataFrame of models.
    """
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame()
    if metric not in comparison_df.columns:
        raise KeyError(f"Metric {metric!r} not found in comparison data.")

    ranked = comparison_df.sort_values(
        by=metric, ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked
