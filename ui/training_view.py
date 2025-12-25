"""
UI component for model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import pickle
import re
from time import perf_counter
from typing import Any
import zipfile

import streamlit as st
import pandas as pd
from app import config
from models import decision_tree, knn, logistic_regression, naive_bayes, svm


def _title_case_df(df):
    """Return a copy with title-cased headers for display."""

    if df is None:
        return df
    return df.rename(columns=lambda c: str(c).replace("_", " ").title())


RANDOM_SEED = 42


def _safe_filename(stem: str) -> str:
    s = "model" if stem is None else str(stem)
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_.-")
    return s or "model"


def _serialize_estimator(estimator: Any) -> tuple[bytes, str, str]:
    """Return (bytes, extension, mime) for a fitted estimator."""

    try:
        import joblib

        buf = BytesIO()
        joblib.dump(estimator, buf)
        return buf.getvalue(), "joblib", "application/octet-stream"
    except (ImportError, ModuleNotFoundError):
        pass
    except (TypeError, ValueError, AttributeError, pickle.PicklingError):

        pass

    data = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
    return data, "pkl", "application/octet-stream"


def _build_models_zip_bytes(trained_models: dict[str, Any]) -> tuple[bytes | None, str | None]:
    """Build a ZIP of all trained models.

    Returns (zip_bytes, error_message). If no models could be serialized, returns (None, msg).
    """

    if not trained_models:
        return None, "No trained models available."

    errors: list[str] = []
    zip_buf = BytesIO()
    written = 0
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        root = "trained_models/"
        for model_key, artifact in trained_models.items():
            estimator = getattr(artifact, "estimator", None)
            if estimator is None:
                errors.append(f"{model_key}: missing estimator")
                continue
            try:
                payload, ext, _mime = _serialize_estimator(estimator)
            except (TypeError, ValueError, AttributeError, pickle.PicklingError) as exc:
                errors.append(
                    f"{model_key}: serialize failed ({type(exc).__name__}: {exc})")
                continue

            fname = f"{_safe_filename(model_key)}_trained.{ext}"
            zf.writestr(root + fname, payload)
            written += 1

        if errors:
            zf.writestr(root + "errors.txt", "\n".join(errors).encode("utf-8"))

    if written == 0:
        return None, "Could not serialize any trained models."
    return zip_buf.getvalue(), None


@dataclass(frozen=True)
class TrainedModelArtifact:
    model_key: str
    params: dict[str, Any]
    train_seconds: float
    estimated_seconds: float | None
    estimator: Any


def _estimate_training_seconds(model_key: str, *, n_samples: int, n_features: int) -> float | None:
    """Very rough heuristics to give the user *some* sense of expected time."""

    n_samples = max(int(n_samples), 1)
    n_features = max(int(n_features), 1)

    if model_key == "Logistic Regression":

        return float(min(30.0, 2.0e-6 * n_samples * n_features * 15))
    if model_key == "K-Nearest Neighbors":

        return float(min(5.0, 2.0e-7 * n_samples * n_features))
    if model_key == "Support Vector Machine":

        return float(min(120.0, 1.0e-7 * (n_samples**2)))
    if model_key == "Decision Tree":
        return float(min(15.0, 2.0e-6 * n_samples * n_features))
    if model_key == "Naive Bayes":
        return float(min(5.0, 5.0e-7 * n_samples * n_features))
    return None


def _build_model(model_key: str):
    """Create a model wrapper with sensible defaults."""

    if model_key == "Logistic Regression":

        return logistic_regression.LogisticRegressionModel(
            params={"random_state": RANDOM_SEED}
        )
    if model_key == "K-Nearest Neighbors":
        return knn.KNNModel(params={})
    if model_key == "Support Vector Machine":

        return svm.SVMModel(params={"probability": False})
    if model_key == "Decision Tree":
        return decision_tree.DecisionTreeModel(params={"random_state": RANDOM_SEED})
    if model_key == "Naive Bayes":
        return naive_bayes.NaiveBayesModel(params={})
    raise KeyError(f"Unsupported model: {model_key}")


def show():
    """Shows the model training page."""
    st.header("5. Model Training")

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
            "Target column has changed since preprocessing. Re-run Preprocessing (Step 4) to keep stages consistent."
        )
        return

    X_train = artifacts.X_train
    y_train = artifacts.y_train

    st.subheader("Training Data")
    st.write(
        {
            "X_train_shape": getattr(X_train, "shape", None),
            "y_train_shape": getattr(y_train, "shape", None),
            "num_features": getattr(X_train, "shape", (None, None))[1],
        }
    )

    selected_models = st.multiselect(
        "Select Models to Train",
        options=config.SUPPORTED_MODELS,
        default=["Logistic Regression"],
    )

    col_a, col_b = st.columns(2)
    with col_a:
        overwrite = st.checkbox(
            "Overwrite previously trained models",
            value=False,
        )
    with col_b:
        clear_all = st.button("Clear trained models")

    if clear_all:
        st.session_state.trained_models = {}
        st.success("Cleared trained models.")
        return

    if not selected_models:
        st.info("Select one or more models to train.")
        return

    n_samples = int(getattr(X_train, "shape", (0, 0))[0] or 0)
    n_features = int(getattr(X_train, "shape", (0, 0))[1] or 0)

    st.subheader("Estimated Training Time")
    estimates = []
    for model_key in selected_models:
        est = _estimate_training_seconds(
            model_key, n_samples=n_samples, n_features=n_features)
        estimates.append(
            {"model": model_key, "estimated_seconds": None if est is None else round(est, 2)})
    st.dataframe(_title_case_df(pd.DataFrame(estimates)), width="stretch")

    if st.button("Train Selected Models"):
        if "trained_models" not in st.session_state or st.session_state.trained_models is None:
            st.session_state.trained_models = {}

        overall = st.progress(0)
        total = len(selected_models)

        for idx, model_key in enumerate(selected_models, start=1):
            if (not overwrite) and (model_key in st.session_state.trained_models):
                st.info(f"Skipping {model_key} (already trained).")
                overall.progress(int((idx / total) * 100))
                continue

            box = st.container(border=True)
            with box:
                st.write(f"**{model_key}**")

                est_seconds = _estimate_training_seconds(
                    model_key,
                    n_samples=n_samples,
                    n_features=n_features,
                )
                if est_seconds is None:
                    st.caption("Estimated training time: unavailable")
                else:
                    st.caption(f"Estimated training time: ~{est_seconds:.2f}s")

                try:
                    model = _build_model(model_key)
                except (KeyError, TypeError, ValueError) as exc:
                    st.error(f"Failed to initialize model: {exc}")
                    overall.progress(int((idx / total) * 100))
                    continue

                start = perf_counter()
                try:
                    model.train(X_train, y_train)
                except (ValueError, TypeError, RuntimeError, MemoryError) as exc:
                    st.error(f"Training failed: {exc}")
                    overall.progress(int((idx / total) * 100))
                    continue
                train_seconds = perf_counter() - start

                artifact = TrainedModelArtifact(
                    model_key=model_key,
                    params=model.get_params(),
                    train_seconds=float(train_seconds),
                    estimated_seconds=None if est_seconds is None else float(
                        est_seconds),
                    estimator=model.get_estimator(),
                )

                st.session_state.trained_models[model_key] = artifact
                st.success(f"Trained in {train_seconds:.3f}s")

            overall.progress(int((idx / total) * 100))

        st.success("Training run complete.")

    trained = st.session_state.get("trained_models", {})
    if trained:
        st.subheader("Trained Models")
        rows = []
        for key, artifact in trained.items():
            rows.append(
                {
                    "model": key,
                    "train_seconds": round(float(getattr(artifact, "train_seconds", 0.0)), 4),
                    "estimated_seconds": None
                    if getattr(artifact, "estimated_seconds", None) is None
                    else round(float(getattr(artifact, "estimated_seconds")), 2),
                }
            )
        st.dataframe(_title_case_df(pd.DataFrame(rows)), width="stretch")

        st.subheader("Download Trained Model")
        model_keys = sorted([str(k) for k in trained.keys()])
        selected_key = st.selectbox(
            "Select a trained model to download",
            options=model_keys,
            index=0,
        )
        selected_artifact = trained.get(selected_key)
        estimator = getattr(selected_artifact, "estimator", None)
        if estimator is None:
            st.warning("Selected model has no estimator to download.")
        else:
            try:
                payload, ext, mime = _serialize_estimator(estimator)
            except (TypeError, ValueError, AttributeError, pickle.PicklingError) as exc:
                st.error(f"Failed to serialize model: {exc}")
            else:
                fname = f"{_safe_filename(selected_key)}_trained.{ext}"
                st.download_button(
                    "Download selected model",
                    data=payload,
                    file_name=fname,
                    mime=mime,
                )

        zip_bytes, zip_err = _build_models_zip_bytes(trained)
        if zip_err is not None:
            st.info(zip_err)
        elif zip_bytes is not None:
            st.download_button(
                "Download all trained models (ZIP)",
                data=zip_bytes,
                file_name="trained_models.zip",
                mime="application/zip",
            )
