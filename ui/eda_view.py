"""
UI component for displaying EDA results.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st
import pandas as pd
from eda import eda_runner
from eda.missing_values import missing_values_table
from issues.datatype_mismatch import detect_datatype_mismatch
from issues.high_cardinality import detect_high_cardinality
from issues.imbalance import detect_imbalance
from issues.thresholds import estimate_issue_detection_thresholds


def _title_case_df(df):
    """Return a copy with title-cased column headers for display."""

    if df is None:
        return None
    return df.rename(columns=lambda c: str(c).replace("_", " ").title())


def show():
    """Shows the EDA results page."""
    st.header("3. Exploratory Data Analysis (EDA)")
    if st.session_state.dataset is not None:
        mapped = st.session_state.get("dataset_mapped")
        df = mapped if mapped is not None else st.session_state.dataset
        target_column = st.session_state.get("target_column")

        column_types = st.session_state.get("column_types") or {}
        if column_types:
            numeric_cols = [
                col
                for col, ctype in column_types.items()
                if ctype == "numeric" and col in df.columns and col != target_column
            ]
            categorical_cols = [
                col
                for col, ctype in column_types.items()
                if ctype == "categorical" and col in df.columns and col != target_column
            ]
        else:
            numeric_cols, categorical_cols = eda_runner.detect_column_types(
                df, target_column=target_column if target_column in df.columns else None
            )

        excluded = {target_column} if target_column in df.columns else set()

        with st.expander("EDA Settings", expanded=True):
            outlier_method = st.selectbox("Outlier method", options=[
                                          "iqr", "zscore"], index=0)
            iqr_multiplier = st.slider(
                "IQR multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
            zscore_threshold = st.slider(
                "Z-score threshold", min_value=2.0, max_value=5.0, value=3.0, step=0.1)
            max_corr_features = st.number_input(
                "Max numeric features for correlation heatmap",
                min_value=2,
                max_value=100,
                value=30,
                step=1,
            )

            dist_cols = st.multiselect(
                "Columns for histograms (numeric only)",
                options=numeric_cols,
                default=numeric_cols[: min(6, len(numeric_cols))],
            )
            cat_bar_cols = st.multiselect(
                "Columns for categorical bar charts (incl. boolean)",
                options=categorical_cols,
                default=categorical_cols[: min(6, len(categorical_cols))],
                help="Plots value counts for selected categorical or boolean columns.",
            )

        if st.button("Run EDA"):
            st.session_state.eda_results = eda_runner.run_eda(
                df,
                target_column=target_column if target_column in df.columns else None,
                outlier_method=outlier_method,
                iqr_multiplier=iqr_multiplier,
                zscore_threshold=zscore_threshold,
                max_corr_features=int(max_corr_features),
                distribution_columns=dist_cols,
                column_types=column_types if column_types else None,
            )
            st.success("EDA complete!")

        results = st.session_state.eda_results
        if results:
            for w in results.get("warnings", []):
                st.warning(w)

            st.subheader("Missing Values")
            missing = results.get("missing_values", {})
            missing_table = missing.get("table")
            if missing_table is not None:
                st.dataframe(_title_case_df(missing_table), width="stretch")

            st.subheader("Correlations (Numerical Features)")
            corr = results.get("correlations", {}).get("matrix")
            if corr is not None and getattr(corr, "shape", (0, 0))[0] >= 2:
                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu",
                    zmin=-1,
                    zmax=1,
                    title="Correlation Heatmap",
                )
                fig.update_traces(colorbar=dict(title="corr"))
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Not enough numeric columns to compute correlations.")

            st.subheader("Feature Distributions (Histograms)")
            if dist_cols:
                for col in dist_cols:
                    fig = px.histogram(
                        df,
                        x=col,
                        nbins=30,
                        title=f"Histogram: {col}",
                    )
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("Select at least one numeric column to show histograms.")

            st.subheader("Categorical Distributions (Bar Charts)")
            if cat_bar_cols:
                for col in cat_bar_cols:
                    vc = df[col].value_counts(dropna=False).reset_index()
                    vc.columns = ["value", "count"]
                    vc["value"] = vc["value"].astype(str)
                    fig = px.bar(
                        vc,
                        x="value",
                        y="count",
                        title=f"Value Counts: {col}",
                    )
                    fig.update_layout(xaxis_title=col, yaxis_title="count")
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info(
                    "Select at least one categorical or boolean column to show bar charts.")

            st.subheader("Outliers (Summary)")
            outliers = results.get("outliers", {})
            outlier_table = outliers.get("table")
            if outlier_table is not None:
                st.dataframe(_title_case_df(outlier_table), width="stretch")
    else:
        st.warning("Please upload a dataset first.")


def show_issues():
    """Shows the Issue Detection page (read-only; no auto-fixing)."""

    st.header("2. Issue Detection")

    if st.session_state.dataset is None:
        st.warning("Please upload a dataset first.")
        return

    mapped = st.session_state.get("dataset_mapped")
    df = mapped if mapped is not None else st.session_state.dataset

    target_column = st.session_state.get("target_column")
    if target_column not in df.columns:
        st.warning("Select a valid target column in the Upload page first.")
        return
    target_column = str(target_column)

    column_types = st.session_state.get("column_types") or {}

    def _convertible_fracs(frame: pd.DataFrame) -> pd.Series:
        """Estimate per-column numeric convertibility fraction."""

        if frame is None or frame.empty:
            return pd.Series(dtype=float)

        fracs: dict[str, float] = {}
        for col in frame.columns:
            series = frame[col].dropna()
            if series.empty:
                fracs[col] = 0.0
                continue
            numeric_conv = pd.to_numeric(series.astype(str), errors="coerce")
            fracs[col] = float(numeric_conv.notna().mean())
        return pd.Series(fracs)

    defaults = estimate_issue_detection_thresholds(
        df,
        target_column=target_column,
        convertible_fracs=_convertible_fracs(df),
        column_types=column_types if column_types else None,
    )

    with st.expander("Issue Detection Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            min_class_prop = st.slider(
                "Min class proportion threshold",
                min_value=0.01,
                max_value=0.50,
                value=float(defaults.get("min_class_proportion", 0.20)),
                step=0.01,
            )
            imbalance_ratio_threshold = st.slider(
                "Imbalance ratio threshold (min/max)",
                min_value=0.01,
                max_value=1.00,
                value=float(defaults.get("imbalance_ratio", 0.50)),
                step=0.01,
            )
            missing_percent_threshold = st.slider(
                "High missingness threshold (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults.get("missingness_threshold_pct", 30.0)),
                step=0.5,
            )

        with col2:
            high_cardinality_threshold = st.number_input(
                "High-cardinality threshold (unique values)",
                min_value=2,
                max_value=10000,
                value=int(defaults.get("high_cardinality_threshold", 50)),
                step=1,
            )
            convertible_fraction_threshold = st.slider(
                "Datatype mismatch threshold (convertible fraction)",
                min_value=0.50,
                max_value=1.00,
                value=float(defaults.get("datatype_mismatch_threshold", 0.90)),
                step=0.01,
            )

    if st.button("Run Issue Detection"):
        imbalance = detect_imbalance(
            df,
            target_column,
            min_class_proportion_threshold=float(min_class_prop),
            imbalance_ratio_threshold=float(imbalance_ratio_threshold),
        )
        high_card = detect_high_cardinality(
            df,
            column_types=column_types if column_types else None,
            target_column=target_column,
            threshold=int(high_cardinality_threshold),
        )
        mismatch = detect_datatype_mismatch(
            df,
            column_types=column_types if column_types else None,
            target_column=target_column,
            convertible_fraction_threshold=float(
                convertible_fraction_threshold),
        )

        missing_tbl = missing_values_table(df)
        high_missing = missing_tbl[missing_tbl["missing_percent"] >= float(
            missing_percent_threshold)].copy()

        st.session_state.issues_results = {
            "imbalance": imbalance,
            "high_cardinality": high_card,
            "datatype_mismatch": mismatch,
            "high_missingness": {
                "threshold_percent": float(missing_percent_threshold),
                "table": high_missing,
            },
        }
        st.success("Issue detection complete!")

    results = st.session_state.get("issues_results")
    if not results:
        st.info("Click 'Run Issue Detection' to analyze the dataset.")
        return

    st.subheader("Class Imbalance")
    imbalance = results.get("imbalance", {})
    dist = imbalance.get("distribution")
    if dist is not None and getattr(dist, "empty", True) is False:
        missing_target_count = int(
            imbalance.get("missing_target_count", 0) or 0)
        if missing_target_count > 0:
            st.info(
                f"Ignored {missing_target_count} null target values in imbalance calculation.")
        st.dataframe(_title_case_df(dist), width="stretch")
        fig = px.bar(dist, x="class", y="count",
                     title="Target Class Distribution")
        st.plotly_chart(fig, width="stretch")
        if imbalance.get("is_imbalanced"):
            st.warning(
                "Class imbalance detected. Consider using stratified splits, class weights, or resampling (later steps)."
            )
        else:
            st.success(
                "No strong class imbalance detected with current thresholds.")
    else:
        st.info("No target distribution available.")

    st.subheader("High Missingness")
    high_missing = results.get("high_missingness", {}).get("table")
    if high_missing is not None and getattr(high_missing, "empty", True) is False:
        st.dataframe(_title_case_df(high_missing), width="stretch")
        fig = px.bar(
            high_missing,
            x="column",
            y="missing_percent",
            title="Columns with High Missingness (%)",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.success("No columns exceed the missingness threshold.")

    st.subheader("High Cardinality Categorical Features")
    high_card = results.get("high_cardinality", {})
    hc_table = high_card.get("table")
    if hc_table is not None and getattr(hc_table, "empty", True) is False:
        st.dataframe(_title_case_df(hc_table), width="stretch")
    else:
        st.success("No high-cardinality categorical columns detected.")

    st.subheader("Data Type Mismatches")
    mismatch = results.get("datatype_mismatch", {})
    mm_table = mismatch.get("table")
    if mm_table is not None and getattr(mm_table, "empty", True) is False:
        st.dataframe(_title_case_df(mm_table), width="stretch")
    else:
        st.success("No datatype mismatch patterns detected.")
