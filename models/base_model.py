"""Model abstraction.

These lightweight wrappers standardize how the UI trains models and stores
artifacts. They intentionally keep sklearn estimators as the underlying
implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Abstract base class for model wrappers."""

    name: str

    @abstractmethod
    def train(self, X, y) -> "BaseModel":
        """Fit the underlying estimator and return self."""

    @abstractmethod
    def predict(self, X):
        """Predict labels."""

    def predict_proba(self, X):
        """Predict class probabilities when supported."""

        estimator: Any = self.get_estimator()
        if not hasattr(estimator, "predict_proba"):
            raise AttributeError(
                "This model does not support predict_proba().")
        return estimator.predict_proba(X)

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get estimator parameters."""

    @abstractmethod
    def set_params(self, **params: Any) -> "BaseModel":
        """Set estimator parameters and return self."""

    @abstractmethod
    def get_estimator(self) -> Any:
        """Return the underlying sklearn estimator."""
