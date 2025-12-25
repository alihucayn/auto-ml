"""
Handles feature scaling.
"""

from __future__ import annotations

from sklearn.preprocessing import StandardScaler, MinMaxScaler


SUPPORTED_SCALERS = {"standard", "minmax", "none"}


def get_scaler(scaler_type: str):
    """
    Returns a scaler instance.

    Args:
        scaler_type: The type of scaler ('standard' or 'minmax').

    Returns:
        A scaler instance.
    """
    scaler_type = str(scaler_type).strip().lower()
    if scaler_type not in SUPPORTED_SCALERS:
        raise ValueError(
            f"Unsupported scaler {scaler_type!r}. Supported: {sorted(SUPPORTED_SCALERS)}")

    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler()
    return None
