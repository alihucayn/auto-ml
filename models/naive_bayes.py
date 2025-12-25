"""
Naive Bayes model implementation.
"""

from __future__ import annotations

from typing import Any

from models.base_model import BaseModel
from sklearn.naive_bayes import GaussianNB


class NaiveBayesModel(BaseModel):
    """Naive Bayes model."""

    name = "Naive Bayes"

    def __init__(self, params=None):
        default_params: dict[str, Any] = {}
        merged = {**default_params, **(params or {})}
        self.model = GaussianNB(**merged)

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
