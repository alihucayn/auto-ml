"""
Compares the performance of multiple models.
"""

from __future__ import annotations

import pandas as pd


def compare_models(model_metrics: dict) -> pd.DataFrame:
    """
    Compares model performance based on metrics.

    Args:
        model_metrics: A dictionary of metrics for each model.

    Returns:
        A DataFrame with the comparison results.
    """
    if not model_metrics:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(model_metrics, orient="index")
    df.index.name = "model"
    df = df.reset_index()
    return df
