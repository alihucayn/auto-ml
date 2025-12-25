"""
UI component for dataset upload.
"""

import pandas as pd
import streamlit as st
from core import data_loader, metadata
from eda import eda_runner
from utils.validators import (
    validate_classification_target,
    validate_csv_upload,
)


def _title_case_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with title-cased headers for display only."""

    if df is None:
        return df
    return df.rename(columns=lambda c: str(c).replace("_", " ").title())


def _compute_dataset_signature(df: pd.DataFrame) -> str:
    """Return a stable signature for the loaded dataset."""

    import hashlib

    sample_csv = df.head(50).to_csv(index=False)
    meta = f"{df.shape}|{list(df.columns)}|{list(map(str, df.dtypes))}|{sample_csv}"
    return hashlib.md5(meta.encode("utf-8", "replace")).hexdigest()


def _infer_column_type(
    df: pd.DataFrame,
    col: str,
    *,
    unique_threshold: int = 5,
    ratio_threshold: float = 0.005,
) -> str:
    """Heuristic column typing with uniqueness guard for numeric columns."""

    series = df[col]

    if pd.api.types.is_numeric_dtype(series):
        uniq = series.nunique(dropna=True)
        total = max(len(series), 1)
        if uniq <= unique_threshold or (uniq / total) <= ratio_threshold:
            return "categorical"
        return "numeric"

    return "categorical"


def _clear_downstream_artifacts() -> None:
    """Clear artifacts that depend on dataset/target selection."""

    st.session_state.eda_results = None
    st.session_state.issues_results = None
    st.session_state.preprocessed_data = None
    st.session_state.trained_models = {}
    st.session_state.optimization_results = None
    st.session_state.model_comparison = None
    st.session_state.report = None


def _clear_column_definitions() -> None:
    """Clear column typing/mapping definitions and related UI state."""

    st.session_state.column_types = None
    st.session_state.column_categories = None
    st.session_state.column_value_maps = None
    st.session_state.dataset_mapped = None


def _apply_value_maps(
    df: pd.DataFrame,
    value_maps: dict[str, dict[str, str]] | None,
    column_types: dict[str, str] | None,
) -> pd.DataFrame:
    """Apply categorical value remapping to a DataFrame copy."""

    if not value_maps:
        return df

    mapped = df.copy()
    for col, mapping in value_maps.items():
        if not mapping or col not in mapped.columns:
            continue
        if column_types and column_types.get(col) != "categorical":
            continue

        mapped[col] = mapped[col].map(lambda v: mapping.get(str(v), v))

    return mapped


def _update_mapped_dataset() -> None:
    """Refresh the mapped dataset based on current value maps and column types."""

    df = st.session_state.get("dataset")
    if df is None:
        st.session_state.dataset_mapped = None
        return

    value_maps = st.session_state.get("column_value_maps") or {}
    column_types = st.session_state.get("column_types") or {}

    if not value_maps:
        st.session_state.dataset_mapped = df
        return

    st.session_state.dataset_mapped = _apply_value_maps(
        df, value_maps, column_types)


def show():
    st.header("1. Upload Dataset")

    def _clear_column_level_state() -> None:
        for key in list(st.session_state.keys()):
            if key.startswith(("type_", "cats_", "cats_extra_", "map_editor_")):
                del st.session_state[key]

    uploaded_file = st.file_uploader("Upload CSV file", type=[
                                     "csv"], accept_multiple_files=False)

    if uploaded_file is not None:
        validation = validate_csv_upload(uploaded_file)
        for msg in validation.warnings:
            st.warning(msg)
        if not validation.ok:
            for err in validation.errors:
                st.error(err)
        else:
            load_result = data_loader.load_csv(uploaded_file)
            for msg in load_result.warnings:
                st.warning(msg)

            if load_result.error:
                st.error(load_result.error)
            elif load_result.df is not None:
                new_df = load_result.df
                new_sig = _compute_dataset_signature(new_df)
                current_sig = st.session_state.get("dataset_signature")

                if current_sig != new_sig:
                    _clear_downstream_artifacts()
                    _clear_column_definitions()
                    _clear_column_level_state()
                    st.session_state.dataset = new_df
                    st.session_state.dataset_signature = new_sig
                    st.session_state.target_column = None
                    _update_mapped_dataset()
                    st.success("Dataset loaded successfully!")
                else:
                    st.info(
                        "Uploaded dataset matches the one already loaded; keeping existing session state.")

    df = st.session_state.get("dataset")
    if df is None:
        st.info("Upload a dataset to configure column types and target.")
        return

    rows, cols = metadata.get_shape(df)
    st.subheader("Dataset Summary")
    summary_df = pd.DataFrame(
        [
            {"Metric": "Rows", "Value": rows},
            {"Metric": "Columns", "Value": cols},
        ]
    )
    st.table(summary_df)

    st.subheader("First 5 Rows")
    st.dataframe(_title_case_df(
        metadata.get_preview(df, n=5)), width="stretch")

    detected_types = st.session_state.get("column_types")
    sig_changed = st.session_state.get(
        "column_types_sig") != st.session_state.get("dataset_signature")
    if detected_types is None or sig_changed:
        type_map = {col: _infer_column_type(df, col) for col in df.columns}
        st.session_state.column_types = type_map
        st.session_state.column_categories = None
        st.session_state.column_value_maps = None
        st.session_state.column_types_sig = st.session_state.get(
            "dataset_signature")

    # Keep mapped dataset in sync with latest maps/types
    _update_mapped_dataset()

    current_types = st.session_state.get("column_types") or {}
    current_maps = st.session_state.get("column_value_maps") or {}

    st.subheader("Confirm Column Types")
    table_df = pd.DataFrame(
        [
            {
                "column": col,
                "type": current_types.get(col, _infer_column_type(df, col)),
            }
            for col in df.columns
        ]
    )
    st.dataframe(_title_case_df(table_df), width="stretch")

    with st.expander("Edit Column Types & Categories", expanded=False):
        st.caption(
            "Adjust misclassified encoded columns and remap categorical values.")

        new_types: dict[str, str] = {}
        new_maps: dict[str, dict[str, str]] = {}

        for col in df.columns:
            default_type = current_types.get(col, _infer_column_type(df, col))
            type_key = f"type_{col}"

            selected_value = st.session_state.get(type_key, default_type)

            col_type = st.selectbox(
                f"Type for {col}",
                options=["numeric", "categorical"],
                index=0 if selected_value == "numeric" else 1,
                key=type_key,
            )
            new_types[col] = col_type

            if col_type == "categorical":
                uniq = df[col].dropna().unique().tolist()
                uniq = [str(v) for v in uniq][:50]

                map_df = pd.DataFrame(
                    {
                        "original": uniq,
                        "mapped": [current_maps.get(col, {}).get(u, u) for u in uniq],
                    }
                )
                edited = st.data_editor(
                    map_df,
                    num_rows="fixed",
                    key=f"map_editor_{col}",
                    width="stretch",
                    column_config={
                        "original": st.column_config.TextColumn(disabled=True),
                        "mapped": st.column_config.TextColumn(help="Enter standardized value for each category"),
                    },
                )
                new_maps[col] = {
                    str(row["original"]): str(row["mapped"]) for _, row in edited.iterrows() if str(row["original"])
                }

        st.session_state.column_types = new_types
        st.session_state.column_categories = None
        st.session_state.column_value_maps = new_maps
        st.session_state.column_types_sig = st.session_state.get(
            "dataset_signature")
        _update_mapped_dataset()
        st.info("Changes apply immediately; no save needed.")

    st.subheader("Target Column")
    categorical_candidates = [col for col, t in (st.session_state.get(
        "column_types") or {}).items() if t == "categorical"]
    previous_target = st.session_state.get("target_column")
    previous_target_str = previous_target if isinstance(
        previous_target, str) else None

    if not categorical_candidates:
        st.error(
            "No categorical columns available for classification target. Adjust column types above.")
        st.session_state.target_column = None
        return

    target_column = st.selectbox(
        "Select the target column (must be categorical)",
        options=categorical_candidates,
        index=(categorical_candidates.index(previous_target_str)
               if previous_target_str in categorical_candidates else 0),
    )
    if previous_target_str is not None and target_column != previous_target_str:
        _clear_downstream_artifacts()

    target_validation = validate_classification_target(df, target_column)
    for w in target_validation.warnings:
        st.warning(w)
    for e in target_validation.errors:
        st.error(e)

    if not target_validation.ok:
        st.session_state.target_column = None
        return

    st.session_state.target_column = target_column

    st.subheader("Target Class Distribution")
    df_for_counts = st.session_state.get("dataset_mapped")
    if df_for_counts is None:
        df_for_counts = df
    try:
        dist = metadata.get_target_class_distribution(
            df_for_counts, target_column, normalize=False, dropna=False)
        st.dataframe(_title_case_df(dist), width="stretch")
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Could not compute class distribution: {exc}")
