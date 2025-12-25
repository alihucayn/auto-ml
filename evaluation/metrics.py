"""
Calculates evaluation metrics for models.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calculates classification metrics.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.

    Returns:
        A dictionary of metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
