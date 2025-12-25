"""
UI component for the preprocessing step.
"""

from __future__ import annotations

from io import BytesIO
import zipfile

import pandas as pd
import streamlit as st
from preprocessing.pipeline import PreprocessingConfig, preprocess_dataset


def _build_preprocessed_csv_bytes(artifacts, *, target_column: str) -> bytes:
    X_all = pd.concat([artifacts.X_train, artifacts.X_test], axis=0)
    y_all = pd.concat([artifacts.y_train, artifacts.y_test], axis=0)

    try:
        X_all = X_all.sort_index()
        y_all = y_all.sort_index()
    except (TypeError, ValueError, KeyError):

        pass

    out = X_all.copy()
    out[str(target_column)] = y_all.reindex(out.index)

    buf = BytesIO()
    out.to_csv(buf, index=False)
    return buf.getvalue()


def _build_preprocessed_split_zip_bytes(artifacts, *, target_column: str) -> bytes:
    def _with_target(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        out = X.copy()
        out[str(target_column)] = y.reindex(out.index)
        return out

    train_df = _with_target(artifacts.X_train, artifacts.y_train)
    test_df = _with_target(artifacts.X_test, artifacts.y_test)

    train_buf = BytesIO()
    test_buf = BytesIO()
    train_df.to_csv(train_buf, index=False)
    test_df.to_csv(test_buf, index=False)

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("preprocessed_train.csv", train_buf.getvalue())
        zf.writestr("preprocessed_test.csv", test_buf.getvalue())
    return zip_buf.getvalue()


def show():
    """Shows the preprocessing page."""
    st.header("4. Preprocessing")
    if st.session_state.dataset is not None:
        mapped = st.session_state.get("dataset_mapped")
        df = mapped if mapped is not None else st.session_state.dataset
        target_column = st.session_state.get("target_column")

        if target_column not in df.columns:
            st.warning("Select a valid target column in the Upload page first.")
            return

        st.subheader("Configuration")
        with st.expander("Preprocessing Settings", expanded=True):
            feature_options = [c for c in df.columns if c != target_column]
            selected_features = st.multiselect(
                "Select features to use for training",
                options=feature_options,
                default=feature_options,
                help="These raw columns will be used for preprocessing and training. Unselected columns are ignored in all downstream steps.",
            )

            test_size = st.slider("Test size", min_value=0.1,
                                  max_value=0.4, value=0.2, step=0.05)
            stratify = st.checkbox("Stratify split by target", value=True)

            col1, col2 = st.columns(2)
            with col1:
                numeric_imputer = st.selectbox(
                    "Numeric imputation",
                    options=["mean", "median", "most_frequent"],
                    index=0,
                )
                encoder_type = st.selectbox("Categorical encoding", options=[
                                            "onehot", "label"], index=0)

            with col2:
                scaler_type = st.selectbox(
                    "Scaling", options=["standard", "minmax", "none"], index=0)
                categorical_imputer = st.selectbox(
                    "Categorical imputation",
                    options=["most_frequent"],
                    index=0,
                )

            st.markdown("---")
            outlier_handling = st.selectbox(
                "Outlier handling (requires consent)",
                options=["none", "cap", "remove"],
                index=0,
                help="Cap: winsorize numeric values using IQR bounds (train-fitted). Remove: drop rows with IQR outliers.",
            )
            consent = True
            if outlier_handling != "none":
                consent = st.checkbox(
                    "I understand this may change the dataset and I consent.",
                    value=False,
                )
            iqr_multiplier = st.slider(
                "IQR multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

        if outlier_handling != "none" and not consent:
            st.info("Enable consent to run outlier handling.")

        if st.button("Run Preprocessing"):
            if outlier_handling != "none" and not consent:
                st.error("Consent required for outlier handling.")
                return

            cfg = PreprocessingConfig(
                target_column=str(target_column),
                feature_columns=[str(c) for c in selected_features],
                test_size=float(test_size),
                stratify=bool(stratify),
                numeric_imputer_strategy=str(numeric_imputer),
                categorical_imputer_strategy=str(categorical_imputer),
                encoder_type=str(encoder_type),
                scaler_type=str(scaler_type),
                outlier_handling=str(outlier_handling),
                outlier_method="iqr",
                iqr_multiplier=float(iqr_multiplier),
            )

            column_types = st.session_state.get("column_types") or {}

            try:
                artifacts = preprocess_dataset(
                    df,
                    config=cfg,
                    column_types=column_types if column_types else None,
                )
            except (ValueError, KeyError, RuntimeError, TypeError) as exc:
                st.error(f"Preprocessing failed: {exc}")
                return

            st.session_state.preprocessed_data = artifacts

            st.session_state.trained_models = {}
            st.session_state.optimization_results = None
            st.session_state.model_comparison = None
            st.session_state.report = None
            st.success("Preprocessing complete!")

        artifacts = st.session_state.get("preprocessed_data")
        if artifacts is not None:
            st.subheader("Output")
            if getattr(artifacts, "removed_outlier_rows", 0) > 0:
                st.info(
                    f"Removed {artifacts.removed_outlier_rows} rows due to outliers.")

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label="Download Preprocessed Train+Test (ZIP)",
                    data=_build_preprocessed_split_zip_bytes(
                        artifacts, target_column=str(target_column)
                    ),
                    file_name="preprocessed_splits.zip",
                    mime="application/zip",
                    width="stretch",
                    help="Downloads `preprocessed_train.csv` and `preprocessed_test.csv`.",
                )

            with dl2:
                st.download_button(
                    label="Download Full Preprocessed Dataset (CSV)",
                    data=_build_preprocessed_csv_bytes(
                        artifacts, target_column=str(target_column)
                    ),
                    file_name="preprocessed_dataset.csv",
                    mime="text/csv",
                    width="stretch",
                    help="Exports transformed features + target by concatenating train + test splits.",
                )

            st.write(
                {
                    "X_train_shape": artifacts.X_train.shape,
                    "X_test_shape": artifacts.X_test.shape,
                    "num_features_after": len(artifacts.feature_names),
                }
            )
            st.subheader("Preview: X_train")
            st.dataframe(artifacts.X_train.head(), width="stretch")
    else:
        st.warning("Please upload a dataset first.")
