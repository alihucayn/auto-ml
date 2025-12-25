"""
Handles CSV dataset loading and basic validation.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CsvLoadResult:
    """Outcome of loading a CSV upload."""

    df: pd.DataFrame | None
    warnings: list[str]
    error: str | None


def _read_csv_with_fallbacks(file_bytes: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Read CSV bytes into a DataFrame using conservative fallbacks.

    Strategy:
    - Try fast C engine first.
    - If it fails (common with malformed CSVs), retry with Python engine and
      `on_bad_lines='skip'` to tolerate corrupted rows.
    - Try a few encodings to handle common decoding issues.

    Returns:
        (df, warnings)
    """

    warnings: list[str] = []
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1"]

                          
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            if encoding != "utf-8":
                warnings.append(f"Decoded CSV using {encoding!r} encoding.")
            return df, warnings
        except UnicodeDecodeError:
            continue
        except Exception:
                                         
            break

                                               
    last_exc: Exception | None = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=encoding,
                engine="python",
                on_bad_lines="skip",
            )
            warnings.append(
                "Some malformed rows may have been skipped while reading the CSV.")
            if encoding != "utf-8":
                warnings.append(f"Decoded CSV using {encoding!r} encoding.")
            return df, warnings
        except Exception as exc:
            last_exc = exc

    raise last_exc or ValueError("Failed to parse CSV.")


def load_csv(uploaded_file) -> CsvLoadResult:
    """Load a Streamlit-uploaded CSV into a DataFrame.

    This function is intentionally UI-agnostic (no Streamlit calls). The UI layer
    should decide how to present errors/warnings.

    Args:
        uploaded_file: Streamlit UploadedFile (or similar). Must support `.getvalue()`.

    Returns:
        CsvLoadResult containing the DataFrame, warnings, and optional error.
    """

    if uploaded_file is None:
        return CsvLoadResult(df=None, warnings=[], error="No file uploaded.")

    try:
        file_bytes = uploaded_file.getvalue()
    except Exception:
        return CsvLoadResult(df=None, warnings=[], error="Could not read uploaded file bytes.")

    if not file_bytes:
        return CsvLoadResult(df=None, warnings=[], error="Uploaded file is empty.")

    try:
        df, warnings = _read_csv_with_fallbacks(file_bytes)
    except Exception as exc:
        return CsvLoadResult(df=None, warnings=[], error=f"Error loading CSV file: {exc}")

    if df.empty:
        return CsvLoadResult(df=None, warnings=[], error="CSV parsed successfully but contains no rows.")
    if df.columns.size == 0:
        return CsvLoadResult(df=None, warnings=[], error="CSV parsed successfully but contains no columns.")

    return CsvLoadResult(df=df, warnings=warnings, error=None)
