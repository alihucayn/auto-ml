"""UI component for running inference using trained models.

This page allows users to upload new data (CSV) and generate predictions using
models trained in Step 5. The same fitted preprocessing transformer from Step 4
is applied to ensure feature consistency.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.exceptions import NotFittedError


def _coerce_user_value(raw: str, *, is_numeric: bool) -> Any:
    s = "" if raw is None else str(raw).strip()
    if s == "":
        return pd.NA
    if not is_numeric:
        return s
    try:
        return float(s)
    except ValueError:
        return pd.NA


def _get_preprocessor_input_columns(preprocessor: Any) -> list[str] | None:
    cols = getattr(preprocessor, "feature_names_in_", None)
    if cols is not None:
        try:
            return [str(c) for c in list(cols)]
        except (TypeError, ValueError):
            pass

    transformers = getattr(preprocessor, "transformers", None)
    if not transformers:
        return None

    out: list[str] = []
    try:
        for _name, _transformer, columns in transformers:
            if columns is None:
                continue
            if isinstance(columns, (list, tuple)):
                out.extend([str(c) for c in columns])
            else:
                out.append(str(columns))
    except (TypeError, ValueError):
        return None

    seen: set[str] = set()
    ordered: list[str] = []
    for c in out:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered or None


def _apply_value_maps(df_in: pd.DataFrame, value_maps: dict[str, dict[str, str]], column_types: dict[str, str]) -> pd.DataFrame:
    """Apply upload-page value maps to a DataFrame copy."""

    if not value_maps:
        return df_in.copy()

    df = df_in.copy()
    for col, mapping in value_maps.items():
        if not mapping or col not in df.columns:
            continue
        if column_types and column_types.get(col) != "categorical":
            continue
        df[col] = df[col].map(lambda v: mapping.get(str(v), v))
    return df


def show():
    st.header("9. Inference")

    artifacts = st.session_state.get("preprocessed_data")
    if artifacts is None:
        st.warning("Run Preprocessing first (Step 4) to fit the transformer.")
        return

    trained = st.session_state.get("trained_models") or {}
    if not trained:
        st.warning("Train at least one model first (Step 5).")
        return

    preprocessor = getattr(artifacts, "preprocessor", None)
    feature_names = getattr(artifacts, "feature_names", None)
    target_column = getattr(
        getattr(artifacts, "config", None), "target_column", None)

    if preprocessor is None or not feature_names:
        st.error(
            "Missing preprocessing transformer/feature names. Re-run Preprocessing (Step 4).")
        return

    model_key = st.selectbox(
        "Select trained model",
        options=sorted([str(k) for k in trained.keys()]),
        index=0,
    )

    selected_artifact = trained.get(model_key)
    estimator = getattr(selected_artifact, "estimator", None)
    if estimator is None:
        st.error("Selected trained model has no estimator.")
        return

    uploaded = st.file_uploader("Upload CSV for inference", type=[
                                "csv"], accept_multiple_files=False)
    mode = st.radio(
        "Inference mode",
        options=["Batch CSV", "Single sample (manual inputs)"],
        index=0,
        horizontal=True,
    )

    cfg = getattr(artifacts, "config", None)
    cfg_features = getattr(cfg, "feature_columns", None)
    if cfg_features:
        input_cols = [str(c) for c in cfg_features]
    else:
        input_cols = _get_preprocessor_input_columns(preprocessor)

    if not input_cols:
        st.error(
            "Could not determine required raw input feature columns. Re-run Preprocessing (Step 4).")
        return

    st.caption(
        "Using the raw feature columns selected in Preprocessing (Step 4). "
        "Inference-time feature selection is disabled to avoid train/inference mismatch."
    )

    base_df = st.session_state.get("dataset")
    mapped_df = st.session_state.get("dataset_mapped")
    base_dtypes: dict[str, Any] = {}
    column_types = st.session_state.get("column_types") or {}
    value_maps = st.session_state.get("column_value_maps") or {}
    base_categories: dict[str, list[str]] = {}
    category_source = mapped_df if isinstance(
        mapped_df, pd.DataFrame) else base_df
    if isinstance(category_source, pd.DataFrame):
        try:
            base_dtypes = {
                str(c): category_source[c].dtype for c in category_source.columns}
            for c, t in column_types.items():
                if t != "categorical" or c not in category_source.columns:
                    continue
                uniques = category_source[c].dropna().astype(
                    str).unique().tolist()
                # Preserve order, cap to a reasonable size for the dropdown
                seen = set()
                ordered = []
                for v in uniques:
                    if v not in seen:
                        ordered.append(v)
                        seen.add(v)
                    if len(ordered) >= 200:
                        break
                base_categories[c] = ordered
        except (TypeError, KeyError, ValueError):
            base_dtypes = {}

    def _build_X_raw_from_df(df_in: pd.DataFrame) -> pd.DataFrame:
        df_work = _apply_value_maps(df_in, value_maps, column_types)
        if target_column and target_column in df_work.columns:
            df_work = df_work.drop(columns=[target_column])
        out = pd.DataFrame(index=df_work.index)
        for col in input_cols:
            if col in df_work.columns:
                out[col] = df_work[col]
            else:
                out[col] = pd.NA
        return out

    out_index = None

    if mode == "Batch CSV":
        if uploaded is None:
            st.info(
                "Upload a CSV containing the same input feature columns as your training data.")
            return

        try:
            df_in = pd.read_csv(uploaded)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as exc:
            st.error(f"Failed to read CSV: {exc}")
            return

        if df_in.empty:
            st.warning("Uploaded CSV is empty.")
            return

        st.subheader("Input Preview")
        st.dataframe(df_in.head(), width="stretch")

        missing_required = [c for c in input_cols if c not in df_in.columns]
        if missing_required:
            st.warning(
                "Uploaded CSV is missing required feature columns. They will be treated as missing: "
                + ", ".join(sorted(missing_required)[:30])
                + (" …" if len(missing_required) > 30 else "")
            )

        extras = [
            c for c in df_in.columns if c not in input_cols and c != target_column]
        if extras:
            st.caption(
                "Ignoring extra columns not used by preprocessing: "
                + ", ".join([str(c) for c in extras[:30]])
                + (" …" if len(extras) > 30 else "")
            )

        X_raw = _build_X_raw_from_df(df_in)
        out_index = df_in.index
    else:
        st.subheader("Single Sample Inputs")
        st.caption(
            "Enter values for the selected raw feature columns. Leave blank to treat as missing."
        )
        with st.form("single_sample_form"):
            values: dict[str, Any] = {}
            for col in input_cols:
                dtype = base_dtypes.get(col)
                col_type = column_types.get(col)
                is_numeric = bool(
                    dtype is not None and pd.api.types.is_numeric_dtype(dtype))
                is_categorical = col_type == "categorical"

                if is_categorical:
                    options = [""] + base_categories.get(col, [])
                    raw = st.selectbox(
                        label=str(col),
                        options=options,
                        key=f"single_{col}",
                        help="Choose a known category or leave blank to mark missing.",
                    )
                    values[col] = pd.NA if raw == "" else raw
                else:
                    raw = st.text_input(
                        label=str(col),
                        value="",
                        placeholder="numeric" if is_numeric else "text",
                    )
                    values[col] = _coerce_user_value(
                        raw, is_numeric=is_numeric)

            submitted = st.form_submit_button("Run single-sample inference")

        if not submitted:
            return

        row = {col: pd.NA for col in input_cols}
        for col, val in values.items():
            row[col] = val
        X_raw = _build_X_raw_from_df(pd.DataFrame([row]))
        out_index = X_raw.index

    try:
        X_t = preprocessor.transform(X_raw)
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"Preprocessing transform failed: {exc}")
        return

    try:
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        X_df = pd.DataFrame(X_t, columns=list(
            feature_names), index=out_index)
    except (ValueError, TypeError) as exc:
        st.error(f"Failed to build transformed feature frame: {exc}")
        return

    try:
        preds = estimator.predict(X_df)
    except (NotFittedError, ValueError, TypeError, AttributeError) as exc:
        st.error(f"Prediction failed: {exc}")
        return

    out = pd.DataFrame({"prediction": preds}, index=out_index)

    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X_df)
            classes = getattr(estimator, "classes_", None)
            if classes is None:
                classes = list(range(proba.shape[1]))
            for i, cls in enumerate(classes):
                out[f"proba_{str(cls)}"] = proba[:, i]
        except (NotFittedError, ValueError, TypeError, AttributeError):
            pass

    st.subheader("Predictions")
    st.dataframe(out.head(50), width="stretch")

    buf = BytesIO()
    out.to_csv(buf, index=False)
    st.download_button(
        "Download predictions (CSV)",
        data=buf.getvalue(),
        file_name=f"predictions_{str(model_key).replace(' ', '_')}.csv",
        mime="text/csv",
        width="stretch",
    )
