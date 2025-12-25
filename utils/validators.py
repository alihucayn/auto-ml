"""
Utility functions for input validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    """Represents the outcome of an input validation step."""

    ok: bool
    errors: list[str]
    warnings: list[str]


def _get_filename(file) -> str:
    """Best-effort filename extraction for Streamlit's UploadedFile-like objects."""

    name = getattr(file, "name", None)
    return str(name) if name else ""


def _get_size_bytes(file) -> int | None:
    """Best-effort size extraction for Streamlit's UploadedFile-like objects."""

    size = getattr(file, "size", None)
    if isinstance(size, int):
        return size
    return None


def _has_allowed_extension(filename: str, allowed_extensions: Iterable[str]) -> bool:
    filename_lower = filename.lower().strip()
    return any(filename_lower.endswith(ext.lower()) for ext in allowed_extensions)


def validate_csv_upload(
    file,
    *,
    max_size_mb: int = 25,
    allowed_extensions: Iterable[str] = (".csv",),
) -> ValidationResult:
    """Validate a user-uploaded CSV file (basic checks only).

    This function does not attempt to parse the CSV; it only performs safe,
    fast checks (presence, extension, size). Parsing is handled in
    `core.data_loader`.

    Args:
        file: Streamlit UploadedFile (or similar object) from `st.file_uploader`.
        max_size_mb: Maximum allowed upload size in MB.
        allowed_extensions: Allowed filename extensions.

    Returns:
        ValidationResult with `ok`, `errors`, and `warnings`.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if file is None:
        errors.append("No file uploaded.")
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    filename = _get_filename(file)
    if not filename:
        warnings.append("Uploaded file has no filename; continuing anyway.")
    else:
        if not _has_allowed_extension(filename, allowed_extensions):
            errors.append(
                f"Invalid file type. Please upload a CSV file: {allowed_extensions}.")

    size_bytes = _get_size_bytes(file)
    if size_bytes is not None:
        max_bytes = max_size_mb * 1024 * 1024
        if size_bytes <= 0:
            errors.append("Uploaded file is empty.")
        elif size_bytes > max_bytes:
            errors.append(
                f"File too large ({size_bytes / (1024 * 1024):.1f}MB). Max is {max_size_mb}MB.")
    else:
        warnings.append("Could not determine file size; continuing.")

    return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


def _is_integer_like_float(series: pd.Series) -> bool:
    """Return True if all non-null float values are integer-like (e.g., 0.0/1.0)."""

    values = pd.to_numeric(series.dropna(), errors="coerce")
    if values.empty:
        return False
    arr = values.to_numpy(dtype=float, copy=False)
                                              
    return bool(np.isclose(arr, np.round(arr)).all())


def get_classification_target_candidates(
    df: pd.DataFrame,
    *,
    max_classes: int = 200,
) -> list[str]:
    """Return columns that are reasonable classification targets.

    Rules (simple + practical):
    - At least 2 unique non-null values.
    - Object/category/bool always allowed.
    - Integer dtype allowed if class count <= max_classes.
    - Float dtype allowed only if values are integer-like (0.0/1.0 etc.) and class count <= max_classes.
    """

    candidates: list[str] = []
    if df is None or df.empty:
        return candidates

    for col in df.columns:
        s = df[col]
        nunique = int(s.dropna().nunique())
        if nunique < 2:
            continue

        if (
            pd.api.types.is_bool_dtype(s)
            or pd.api.types.is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(s)
        ):
            candidates.append(str(col))
            continue

        if pd.api.types.is_integer_dtype(s):
            if nunique <= int(max_classes):
                candidates.append(str(col))
            continue

        if pd.api.types.is_float_dtype(s):
            if nunique <= int(max_classes) and _is_integer_like_float(s):
                candidates.append(str(col))
            continue

                                                                                   

    return candidates


def validate_classification_target(
    df: pd.DataFrame,
    target_column: str,
    *,
    max_classes: int = 200,
) -> ValidationResult:
    """Validate that the selected target is usable for classification."""

    errors: list[str] = []
    warnings: list[str] = []

    if df is None or df.empty:
        errors.append("No dataset loaded.")
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    if not target_column or target_column not in df.columns:
        errors.append("Select a valid target column.")
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    s = df[target_column]
    nunique = int(s.dropna().nunique())
    if nunique < 2:
        errors.append("Target must have at least 2 classes (unique values).")
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    candidates = set(get_classification_target_candidates(
        df, max_classes=max_classes))
    if target_column not in candidates:
        dtype = str(s.dtype)
        errors.append(
            f"Target column must be categorical for classification. '{target_column}' looks numeric/continuous (dtype={dtype})."
        )
        errors.append(
            "If your classes are encoded as numbers, make sure they are discrete labels like 0/1/2 (not continuous values)."
        )
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    if nunique > 50:
        warnings.append(
            f"Target has {nunique} classes. This may reduce model quality and slow training/metrics."
        )

    return ValidationResult(ok=True, errors=errors, warnings=warnings)
