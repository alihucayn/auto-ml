"""
Grid search for hyperparameter optimization.
"""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


RANDOM_SEED = 42


def run_grid_search(
    model,
    param_grid: dict[str, list[Any]],
    X,
    y,
    *,
    scoring: str = "accuracy",
    cv: int = 5,
    n_jobs: int | None = None,
    verbose: int = 0,
    refit: bool = True,
):
    """
    Performs grid search for a given model.

    Args:
        model: The model to optimize.
        param_grid: The parameter grid.
        X: The feature matrix.
        y: The target vector.

    Returns:
        The fitted GridSearchCV object.
    """

    splitter = StratifiedKFold(n_splits=int(
        cv), shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=splitter,
        n_jobs=n_jobs,
        verbose=int(verbose),
        refit=bool(refit),
        return_train_score=False,
    )
    search.fit(X, y)
    return search
