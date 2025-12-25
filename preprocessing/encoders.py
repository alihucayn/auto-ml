"""
Handles categorical feature encoding.
"""

from __future__ import annotations

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


SUPPORTED_ENCODERS = {"onehot", "label"}


def get_encoder(encoder_type: str):
    """Return a categorical feature encoder.

    Important: `LabelEncoder` is meant for encoding target labels (y), not
    feature columns. For feature "label encoding" we use `OrdinalEncoder`.

    Args:
        encoder_type: "onehot" or "label".

    Returns:
        A scikit-learn transformer suitable for categorical feature matrices.
    """

    encoder_type = str(encoder_type).strip().lower()
    if encoder_type not in SUPPORTED_ENCODERS:
        raise ValueError(
            f"Unsupported encoder {encoder_type!r}. Supported: {sorted(SUPPORTED_ENCODERS)}")

    if encoder_type == "onehot":
                                                                          
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

                                   
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
