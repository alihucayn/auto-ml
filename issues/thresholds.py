"""Estimate data-driven defaults for issue detection thresholds."""

from __future__ import annotations

import math
import pandas as pd


def _safe_quantile(series: pd.Series, q: float) -> float:
    """Return quantile value or NaN when series is empty."""

    if series is None or series.empty:
        return math.nan
    return float(series.quantile(q))


def estimate_issue_detection_thresholds(
    df: pd.DataFrame,
    target_column: str | None = None,
    convertible_fracs: pd.Series | None = None,
    column_types: dict[str, str] | None = None,
) -> dict:
    """Compute dataset-driven defaults for issue detection sliders.

    This function avoids fixed constants (beyond the specified 0.8 minimum for
    dtype conversion) and derives thresholds from the current dataset.
    """

    N = max(len(df), 1)

    missing_rates = df.isna().mean() if not df.empty else pd.Series(dtype=float)
    if column_types:
        categorical_cols = [
            c
            for c, t in column_types.items()
            if t == "categorical" and c in df.columns and c != target_column
        ]
    else:
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)

    if categorical_cols:
        unique_counts = df[categorical_cols].nunique(dropna=True)
        unique_ratios = unique_counts / N
    else:
        unique_ratios = pd.Series(dtype=float)

    if target_column is not None and target_column in df.columns:
        class_counts = df[target_column].value_counts(dropna=False)
        class_props = class_counts / \
            class_counts.sum() if class_counts.sum() else pd.Series(dtype=float)
    else:
        class_props = pd.Series(dtype=float)

    # Rule 1: min class proportion threshold (looser)
    class_quantile = _safe_quantile(class_props, 0.05)
    raw_min_prop = 0.0 if math.isnan(class_quantile) else class_quantile
    min_class_prop = max(raw_min_prop, 2 / N)
    min_class_prop = min(min_class_prop, 0.4)
    min_class_prop = round(min_class_prop, 3)

    # Rule 2: imbalance ratio threshold (looser)
    if class_props.empty or class_props.min() == 0 or class_props.max() == 0:
        raw_ratio = 1.0
    else:
        raw_ratio = float(class_props.min() / class_props.max())

    # Soften the trigger by scaling down the observed ratio and bounding
    imbalance_ratio = raw_ratio * 0.8
    imbalance_ratio = max(min(imbalance_ratio, 0.9), 0.1)
    imbalance_ratio = round(imbalance_ratio, 2)

    # Rule 3: high missingness threshold (%)
    miss_q = _safe_quantile(missing_rates, 0.75)
    miss_mu = float(missing_rates.mean()) if not missing_rates.empty else 0.0
    miss_sigma = float(missing_rates.std()) if not missing_rates.empty else 0.0
    missingness_threshold = min(
        (1.0 if math.isnan(miss_q) else miss_q),
        miss_mu + miss_sigma,
    )
    missingness_threshold_pct = round(missingness_threshold * 100.0, 1)

    # Rule 4: high cardinality threshold (unique values)
    if unique_ratios.empty:
        high_cardinality_threshold = max(2, int(min(50, N)))
    else:
        card_q = _safe_quantile(unique_ratios, 0.90)
        card_ratio = min((0.0 if math.isnan(card_q) else card_q), 0.95)
        high_cardinality_threshold = max(2, int(card_ratio * N))

    # Rule 5: datatype mismatch threshold
    conv_q = _safe_quantile(
        convertible_fracs, 0.10) if convertible_fracs is not None else math.nan
    dtype_mismatch_threshold = max(
        (0.0 if math.isnan(conv_q) else conv_q), 0.8)
    dtype_mismatch_threshold = round(dtype_mismatch_threshold, 2)

    return {
        "min_class_proportion": float(min_class_prop),
        "imbalance_ratio": float(imbalance_ratio),
        "missingness_threshold_pct": float(missingness_threshold_pct),
        "high_cardinality_threshold": int(high_cardinality_threshold),
        "datatype_mismatch_threshold": float(dtype_mismatch_threshold),
    }
