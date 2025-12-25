"""
Logistic Regression model implementation.
"""

from __future__ import annotations

from typing import Any

from models.base_model import BaseModel
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model."""

    name = "Logistic Regression"

    def __init__(self, params=None):
        default_params: dict[str, Any] = {
            "max_iter": 1000,
        }
        merged = {**default_params, **(params or {})}
        self.model = LogisticRegression(**merged)

    def train(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **params: Any):
        self.model.set_params(**params)
        return self

    def get_estimator(self):
        return self.model
