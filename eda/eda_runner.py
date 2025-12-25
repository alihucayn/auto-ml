"""
Orchestrates the Exploratory Data Analysis (EDA) process.
"""

from __future__ import annotations

import pandas as pd

from eda.correlations import analyze_correlations, get_numeric_columns
from eda.distributions import analyze_distributions
from eda.missing_values import analyze_missing_values
from eda.outliers import detect_outliers


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of categorical column names."""

    return df.select_dtypes(exclude="number").columns.tolist()


def detect_column_types(
    df: pd.DataFrame,
    *,
    target_column: str | None = None,
    max_unique_categorical: int = 20,
    unique_fraction_threshold: float = 0.10,
) -> tuple[list[str], list[str]]:
    """Infer numeric vs categorical columns with heuristics for encoded labels.

    - Drops the target column from consideration.
    - Numeric columns with very low unique values (e.g., 0/1 flags) are treated as categorical.
    """

    excluded: set[str] = set()
    if target_column is not None:
        excluded.add(str(target_column))

    n_rows = len(df)
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        if col in excluded:
            continue

        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            nunique = int(non_null.nunique()) if not non_null.empty else 0
            unique_frac = (nunique / n_rows) if n_rows else 1.0
            is_integer_like = non_null.apply(lambda x: float(
                x).is_integer()).all() if not non_null.empty else True

            if pd.api.types.is_bool_dtype(series):
                categorical_cols.append(col)
                continue

            if is_integer_like and (nunique <= max_unique_categorical or unique_frac <= unique_fraction_threshold):
                categorical_cols.append(col)
                continue

            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def run_eda(
    df: pd.DataFrame,
    *,
    target_column: str | None = None,
    outlier_method: str = "iqr",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
    correlation_method: str = "pearson",
    max_corr_features: int = 30,
    distribution_bins: int = 30,
    distribution_columns: list[str] | None = None,
    column_types: dict[str, str] | None = None,
) -> dict:
    """
    Runs the full EDA pipeline.

    Args:
        df: The input DataFrame.

    Returns:
        A dictionary containing EDA results.
    """
    excluded: set[str] = set()
    if target_column is not None:
        excluded.add(str(target_column))

    feature_df = df.drop(columns=list(excluded), errors="ignore")

    warnings: list[str] = []

    if column_types:
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []

        for col in df.columns:
            if col in excluded:
                continue
            ctype = column_types.get(col)
            if ctype == "numeric":
                numeric_cols.append(col)
            elif ctype == "categorical":
                categorical_cols.append(col)

        remaining = [
            c
            for c in df.columns
            if c not in excluded and c not in numeric_cols and c not in categorical_cols
        ]
        if remaining:
            auto_num, auto_cat = detect_column_types(
                df, target_column=target_column)
            for col in auto_num:
                if col in remaining:
                    numeric_cols.append(col)
            for col in auto_cat:
                if col in remaining:
                    categorical_cols.append(col)
    else:
        numeric_cols, categorical_cols = detect_column_types(
            df, target_column=target_column)

    missing = analyze_missing_values(df)
    corr = analyze_correlations(
        feature_df[numeric_cols] if numeric_cols else feature_df,
        method=correlation_method,
        max_features=max_corr_features,
    )

    outliers = detect_outliers(
        feature_df[numeric_cols] if numeric_cols else feature_df,
        method=outlier_method,
        iqr_multiplier=iqr_multiplier,
        zscore_threshold=zscore_threshold,
    )
    warnings.extend(outliers.get("warnings", []))

    distributions = analyze_distributions(
        feature_df,
        columns=[
            c
            for c in (distribution_columns or [])
            if c not in excluded and c in feature_df.columns and c in numeric_cols
        ] if distribution_columns is not None else numeric_cols,
        categorical_columns=[
            c for c in categorical_cols if c not in excluded and c in feature_df.columns
        ],
        bins=distribution_bins,
    )
    warnings.extend(distributions.get("warnings", []))

    if len(numeric_cols) == 0:
        warnings.append(
            "No numeric columns detected; correlation/distribution/outlier views may be limited.")

    return {
        "warnings": warnings,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": missing,
        "correlations": {
            "method": correlation_method,
            "matrix": corr,
        },
        "outliers": outliers,
        "distributions": distributions,
    }
