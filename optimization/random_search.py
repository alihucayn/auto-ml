"""
Random search for hyperparameter optimization.
"""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold


RANDOM_SEED = 42


def run_random_search(
    model,
    param_distributions: dict[str, list[Any]],
    X,
    y,
    *,
    scoring: str = "accuracy",
    cv: int = 5,
    n_iter: int = 20,
    n_jobs: int | None = None,
    verbose: int = 0,
    refit: bool = True,
    random_state: int = RANDOM_SEED,
):
    """
    Performs random search for a given model.

    Args:
        model: The model to optimize.
        param_distributions: The parameter distributions.
        X: The feature matrix.
        y: The target vector.

    Returns:
        The fitted RandomizedSearchCV object.
    """

    splitter = StratifiedKFold(n_splits=int(
        cv), shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        scoring=scoring,
        cv=splitter,
        n_iter=int(n_iter),
        n_jobs=n_jobs,
        verbose=int(verbose),
        refit=bool(refit),
        random_state=int(random_state),
        return_train_score=False,
    )
    search.fit(X, y)
    return search
