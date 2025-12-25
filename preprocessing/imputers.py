"""
Handles missing value imputation.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer


SUPPORTED_STRATEGIES = {"mean", "median", "most_frequent"}


def get_imputer(strategy: str = "mean") -> SimpleImputer:
    """Create a `SimpleImputer` for a supported strategy.

    Args:
        strategy: One of: "mean", "median", "most_frequent".

    Returns:
        Configured SimpleImputer.
    """

    strategy = str(strategy).strip().lower()
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported imputation strategy {strategy!r}. Supported: {sorted(SUPPORTED_STRATEGIES)}")
    return SimpleImputer(strategy=strategy)


def get_numeric_imputer(strategy: str = "mean") -> SimpleImputer:
    """Imputer intended for numeric features."""

    return get_imputer(strategy=strategy)


def get_categorical_imputer(strategy: str = "most_frequent") -> SimpleImputer:
    """Imputer intended for categorical features.

    Note: for categorical data, "most_frequent" is the typical choice.
    """

                                                             
    return get_imputer(strategy=strategy)
