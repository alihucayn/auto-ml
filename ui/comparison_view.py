"""
UI component for model comparison.
"""

from __future__ import annotations

from time import perf_counter

import plotly.express as px
import streamlit as st
import pandas as pd

from evaluation.comparison import compare_models
from evaluation.metrics import calculate_metrics
from evaluation.ranking import rank_models


def _title_case_df(df):
    """Return a copy with title-cased headers for display."""

    if df is None:
        return df
    return df.rename(columns=lambda c: str(c).replace("_", " ").title())


def show():
    """Shows the model comparison page."""
    st.header("7. Model Comparison")

    artifacts = st.session_state.get("preprocessed_data")
    if artifacts is None:
        st.warning(
            "Run Preprocessing first (Step 4) to generate train/test splits.")
        return

    upload_target = st.session_state.get("target_column")
    preprocessing_target = getattr(
        getattr(artifacts, "config", None), "target_column", None)
    if upload_target is None:
        st.warning("Select a target column in the Upload page first.")
        return
    if preprocessing_target is not None and preprocessing_target != upload_target:
        st.warning(
            "Target column has changed since preprocessing. Re-run Preprocessing (Step 4) before comparing."
        )
        return

    trained = st.session_state.get("trained_models") or {}
    if not trained:
        st.warning(
            "Train at least one model first (Step 5), optionally optimize (Step 6).")
        return

    X_test = artifacts.X_test
    y_test = artifacts.y_test

    with st.expander("Comparison Settings", expanded=True):
        rank_metric = st.selectbox(
            "Rank models by",
            options=["accuracy", "f1_score", "precision", "recall"],
            index=0,
        )

    if st.button("Run Model Comparison"):
        metrics_by_model = {}

        for model_key, artifact in trained.items():
            estimator = getattr(artifact, "estimator", None)
            if estimator is None:
                continue

            start = perf_counter()
            try:
                y_pred = estimator.predict(X_test)
            except (ValueError, TypeError, RuntimeError, MemoryError) as exc:
                metrics_by_model[model_key] = {
                    "error": str(exc),
                }
                continue
            pred_seconds = perf_counter() - start

            m = calculate_metrics(y_test, y_pred)
            m["predict_seconds"] = float(pred_seconds)
            m["train_seconds"] = float(
                getattr(artifact, "train_seconds", 0.0) or 0.0)
            metrics_by_model[model_key] = m

        comparison_df = compare_models(metrics_by_model)

        if not comparison_df.empty and "accuracy" in comparison_df.columns:
            ranked_df = rank_models(comparison_df, metric=rank_metric)
        else:
            ranked_df = comparison_df

        st.session_state.model_comparison = {
            "rank_metric": rank_metric,
            "table": ranked_df,
        }
        st.success("Comparison complete!")

    results = st.session_state.get("model_comparison")
    if not results:
        st.info("Click 'Run Model Comparison' to evaluate and rank trained models.")
        return

    table = results.get("table")
    if table is None or getattr(table, "empty", True):
        st.warning("No comparison results available.")
        return

    st.subheader("Results")
    st.dataframe(_title_case_df(table), width="stretch")

    if all(col in table.columns for col in ["model", "accuracy"]):
        st.subheader("Accuracy")
        fig = px.bar(table, x="model", y="accuracy", title="Model Accuracy")
        st.plotly_chart(fig, width="stretch")

    if all(col in table.columns for col in ["model", "f1_score"]):
        st.subheader("F1 (weighted)")
        fig = px.bar(table, x="model", y="f1_score",
                     title="Model F1 Score (weighted)")
        st.plotly_chart(fig, width="stretch")
