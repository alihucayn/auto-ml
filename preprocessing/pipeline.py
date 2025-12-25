"""Preprocessing pipeline construction.

This module builds an sklearn-compatible preprocessing pipeline for tabular
classification data:

- Missing value imputation
- Categorical encoding (OneHot or Ordinal/"label")
- Feature scaling (optional)
- Optional outlier handling (cap/remove) with explicit user consent

The pipeline is designed to be UI-agnostic; Streamlit pages should pass user
choices into these functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing.encoders import get_encoder
from preprocessing.imputers import get_categorical_imputer, get_numeric_imputer
from preprocessing.scalers import get_scaler


RANDOM_SEED = 42


@dataclass(frozen=True)
class PreprocessingConfig:
    """User-selected preprocessing configuration."""

    target_column: str
    feature_columns: list[str] | None = None
    test_size: float = 0.2
    stratify: bool = True

    numeric_imputer_strategy: str = "mean"
    categorical_imputer_strategy: str = "most_frequent"
    encoder_type: str = "onehot"
    scaler_type: str = "standard"

    outlier_handling: str = "none"
    outlier_method: str = "iqr"
    iqr_multiplier: float = 1.5


@dataclass(frozen=True)
class PreprocessedData:
    """Artifacts produced by preprocessing."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: list[str]
    config: PreprocessingConfig
    removed_outlier_rows: int = 0


class IQRCapper(BaseEstimator, TransformerMixin):
    """Cap numeric features using IQR bounds learned during fit.

    Expects a 2D numpy array.
    """

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = float(multiplier)
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, X, y=None):
        del y
        X_arr = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X_arr, 25, axis=0)
        q3 = np.nanpercentile(X_arr, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - (self.multiplier * iqr)
        self.upper_ = q3 + (self.multiplier * iqr)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError(
                "IQRCapper must be fitted before calling transform().")
        return np.clip(X_arr, self.lower_, self.upper_)


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split a full dataset into X and y."""

    if target_column not in df.columns:
        raise KeyError(f"Target column {target_column!r} not found.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature columns."""

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def resolve_feature_types(
    X: pd.DataFrame,
    column_types: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve feature types using provided map, falling back to inference."""

    if not column_types:
        return infer_feature_types(X)

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in X.columns:
        ctype = column_types.get(col)
        if ctype == "numeric":
            numeric_cols.append(col)
        elif ctype == "categorical":
            categorical_cols.append(col)

    remaining = [
        c for c in X.columns if c not in numeric_cols and c not in categorical_cols]
    if remaining:
        auto_num, auto_cat = infer_feature_types(X[remaining])
        numeric_cols.extend(auto_num)
        categorical_cols.extend(auto_cat)

    return numeric_cols, categorical_cols


def remove_outliers_iqr(
    df: pd.DataFrame,
    *,
    numeric_columns: list[str],
    multiplier: float = 1.5,
) -> tuple[pd.DataFrame, int]:
    """Remove rows containing IQR outliers in any numeric column.

    This is a *row-level* operation and therefore cannot be part of an sklearn
    pipeline. Only use it when the user explicitly consents.
    """

    if not numeric_columns:
        return df, 0

    numeric = df[numeric_columns]
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - (float(multiplier) * iqr)
    upper = q3 + (float(multiplier) * iqr)

    within = (numeric.ge(lower) & numeric.le(upper)) | numeric.isna()
    keep_mask = within.all(axis=1)

    removed = int((~keep_mask).sum())
    return df.loc[keep_mask].copy(), removed


def build_preprocessor(
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    config: PreprocessingConfig,
) -> ColumnTransformer:
    """Build an sklearn ColumnTransformer based on configuration."""

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", get_numeric_imputer(config.numeric_imputer_strategy)),
    ]
    if config.outlier_handling == "cap" and config.outlier_method == "iqr":
        numeric_steps.append(
            ("outlier_cap", IQRCapper(multiplier=config.iqr_multiplier)))

    scaler = get_scaler(config.scaler_type)
    if scaler is not None:
        numeric_steps.append(("scaler", scaler))

    num_pipe = Pipeline(steps=numeric_steps)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", get_categorical_imputer(
                config.categorical_imputer_strategy)),
            ("encoder", get_encoder(config.encoder_type)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_columns),
            ("cat", cat_pipe, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def _to_feature_dataframe(
    X_transformed,
    *,
    feature_names: list[str],
    index,
) -> pd.DataFrame:
    """Convert transformed matrix to a DataFrame with feature names."""

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    arr = np.asarray(X_transformed)
    return pd.DataFrame(arr, columns=feature_names, index=index)


def preprocess_dataset(
    df: pd.DataFrame,
    *,
    config: PreprocessingConfig,
    column_types: dict[str, str] | None = None,
) -> PreprocessedData:
    """Run preprocessing end-to-end: optional outlier removal, split, fit, transform.

    Returns artifacts suitable for model training/evaluation.
    """

    working_df = df.copy()
    X, y = split_features_target(working_df, config.target_column)

    if config.feature_columns:
        wanted = [str(c) for c in config.feature_columns]
        missing = [c for c in wanted if c not in X.columns]
        if missing:
            raise KeyError(
                "Selected feature columns not found in dataset: " +
                ", ".join(missing[:30])
                + (" …" if len(missing) > 30 else "")
            )
        X = X[wanted].copy()

    numeric_cols, categorical_cols = resolve_feature_types(
        X, column_types=column_types)

    removed_outlier_rows = 0
    if config.outlier_handling == "remove" and config.outlier_method == "iqr":
        filtered_df, removed_outlier_rows = remove_outliers_iqr(
            working_df,
            numeric_columns=numeric_cols,
            multiplier=config.iqr_multiplier,
        )
        X, y = split_features_target(filtered_df, config.target_column)
        if config.feature_columns:
            X = X[[str(c) for c in config.feature_columns]].copy()
        numeric_cols, categorical_cols = resolve_feature_types(
            X, column_types=column_types)

    stratify_y = y if config.stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(config.test_size),
        random_state=RANDOM_SEED,
        stratify=stratify_y,
    )

    preprocessor = build_preprocessor(
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        config=config,
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except (AttributeError, ValueError, TypeError):

        feature_names = [f"f{i}" for i in range(
            np.asarray(X_train_t).shape[1])]

    X_train_df = _to_feature_dataframe(
        X_train_t, feature_names=feature_names, index=X_train.index)
    X_test_df = _to_feature_dataframe(
        X_test_t, feature_names=feature_names, index=X_test.index)

    return PreprocessedData(
        X_train=X_train_df,
        X_test=X_test_df,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
        config=config,
        removed_outlier_rows=removed_outlier_rows,
    )
