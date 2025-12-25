"""UI component for hyperparameter optimization (Step 6).

This page performs small-scope hyperparameter search over supported classical
classifiers using scikit-learn's CV search utilities.

Design notes:
- Uses ONLY the preprocessed training split to avoid test leakage.
- Stores results in Streamlit session state for later steps.
"""

from __future__ import annotations

from io import BytesIO
import pickle
import re
from time import perf_counter
from typing import Any
import zipfile

import pandas as pd
import streamlit as st

from app import config
from models import decision_tree, knn, logistic_regression, naive_bayes, svm
from optimization.grid_search import run_grid_search
from optimization.random_search import run_random_search
from ui.training_view import TrainedModelArtifact


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


def _build_models_zip_bytes(optimization_results: dict[str, Any]) -> tuple[bytes | None, str | None]:
    """Build a ZIP of all optimized models (best_estimator).

    Returns (zip_bytes, error_message). If no models could be serialized, returns (None, msg).
    """

    if not optimization_results:
        return None, "No optimization results available."

    errors: list[str] = []
    zip_buf = BytesIO()
    written = 0
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        root = "optimized_models/"
        for model_key, res in optimization_results.items():
            estimator = (res or {}).get("best_estimator")
            if estimator is None:
                errors.append(f"{model_key}: missing best_estimator")
                continue
            try:
                payload, ext, _mime = _serialize_estimator(estimator)
            except (TypeError, ValueError, AttributeError, pickle.PicklingError) as exc:
                errors.append(
                    f"{model_key}: serialize failed ({type(exc).__name__}: {exc})")
                continue

            fname = f"{_safe_filename(model_key)}_optimized.{ext}"
            zf.writestr(root + fname, payload)
            written += 1

        if errors:
            zf.writestr(root + "errors.txt", "\n".join(errors).encode("utf-8"))

    if written == 0:
        return None, "Could not serialize any optimized models."
    return zip_buf.getvalue(), None


def _default_estimator(model_key: str):
    if model_key == "Logistic Regression":
        return logistic_regression.LogisticRegressionModel(
            params={"random_state": RANDOM_SEED}
        ).get_estimator()
    if model_key == "K-Nearest Neighbors":
        return knn.KNNModel(params={}).get_estimator()
    if model_key == "Support Vector Machine":
        return svm.SVMModel(params={"probability": False}).get_estimator()
    if model_key == "Decision Tree":
        return decision_tree.DecisionTreeModel(
            params={"random_state": RANDOM_SEED}
        ).get_estimator()
    if model_key == "Naive Bayes":
        return naive_bayes.NaiveBayesModel(params={}).get_estimator()
    raise KeyError(f"Unsupported model: {model_key}")


def _param_space(model_key: str) -> dict[str, list[Any]]:
    """Small, safe search spaces suitable for ≤10k rows."""

    if model_key == "Logistic Regression":
        return {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs"],
            "class_weight": [None, "balanced"],
        }
    if model_key == "K-Nearest Neighbors":
        return {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        }
    if model_key == "Support Vector Machine":
        return {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
            "class_weight": [None, "balanced"],
        }
    if model_key == "Decision Tree":
        return {
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
        }
    if model_key == "Naive Bayes":
        return {
            "var_smoothing": [1e-9, 1e-8, 1e-7],
        }
    raise KeyError(f"Unsupported model: {model_key}")


def show():
    st.header("6. Hyperparameter Optimization")

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
            "Target column has changed since preprocessing. Re-run Preprocessing (Step 4) before optimizing."
        )
        return

    X_train = artifacts.X_train
    y_train = artifacts.y_train

    st.subheader("Inputs")
    st.write(
        {
            "X_train_shape": getattr(X_train, "shape", None),
            "y_train_shape": getattr(y_train, "shape", None),
        }
    )

    with st.expander("Optimization Settings", expanded=True):
        selected_models = st.multiselect(
            "Models to optimize",
            options=config.SUPPORTED_MODELS,
            default=["Logistic Regression"],
        )

        method = st.selectbox("Search method", options=[
                              "grid", "random"], index=0)

        scoring = st.selectbox(
            "Scoring metric",
            options=["accuracy", "f1_weighted",
                     "precision_weighted", "recall_weighted"],
            index=0,
        )

        cv_folds = st.slider("CV folds", min_value=2,
                             max_value=10, value=5, step=1)

        n_iter = 20
        if method == "random":
            n_iter = st.slider("Random search iterations",
                               min_value=5, max_value=50, value=20, step=1)

        use_all_cores = st.checkbox(
            "Use all CPU cores (n_jobs=-1)", value=True)
        n_jobs = -1 if use_all_cores else 1

        replace_trained = st.checkbox(
            "Replace trained estimators with optimized estimators",
            value=False,
            help="If enabled, optimized models overwrite entries in Step 5's trained models for later Comparison.",
        )

    if not selected_models:
        st.info("Select one or more models to optimize.")
        return

    if st.button("Run Optimization"):
        results: dict[str, Any] = {}
        progress = st.progress(0)
        total = len(selected_models)

        for idx, model_key in enumerate(selected_models, start=1):
            box = st.container(border=True)
            with box:
                st.write(f"**{model_key}**")

                try:
                    estimator = _default_estimator(model_key)
                    space = _param_space(model_key)
                except (KeyError, TypeError, ValueError) as exc:
                    st.error(f"Initialization failed: {exc}")
                    progress.progress(int((idx / total) * 100))
                    continue

                start = perf_counter()
                try:
                    if method == "grid":
                        search = run_grid_search(
                            estimator,
                            space,
                            X_train,
                            y_train,
                            scoring=scoring,
                            cv=int(cv_folds),
                            n_jobs=int(n_jobs),
                            verbose=0,
                        )
                    else:
                        search = run_random_search(
                            estimator,
                            space,
                            X_train,
                            y_train,
                            scoring=scoring,
                            cv=int(cv_folds),
                            n_iter=int(n_iter),
                            n_jobs=int(n_jobs),
                            verbose=0,
                            random_state=RANDOM_SEED,
                        )
                except (ValueError, TypeError, RuntimeError, MemoryError) as exc:
                    st.error(f"Optimization failed: {exc}")
                    progress.progress(int((idx / total) * 100))
                    continue

                elapsed = perf_counter() - start

                best_params = getattr(search, "best_params_", {})
                best_score = float(
                    getattr(search, "best_score_", float("nan")))
                best_estimator = getattr(search, "best_estimator_", None)

                results[model_key] = {
                    "method": method,
                    "scoring": scoring,
                    "cv_folds": int(cv_folds),
                    "n_iter": int(n_iter) if method == "random" else None,
                    "best_score": best_score,
                    "best_params": best_params,
                    "elapsed_seconds": float(elapsed),
                    "best_estimator": best_estimator,
                }

                st.success(
                    f"Best {scoring}: {best_score:.4f} (elapsed {elapsed:.2f}s)")
                st.dataframe(pd.DataFrame(
                    [best_params]), width="stretch")

                if replace_trained and best_estimator is not None:
                    if "trained_models" not in st.session_state or st.session_state.trained_models is None:
                        st.session_state.trained_models = {}
                    st.session_state.trained_models[model_key] = TrainedModelArtifact(
                        model_key=model_key,
                        params=best_params,
                        train_seconds=float(elapsed),
                        estimated_seconds=None,
                        estimator=best_estimator,
                    )

            progress.progress(int((idx / total) * 100))

        st.session_state.optimization_results = results
        st.success("Optimization complete!")

    existing = st.session_state.get("optimization_results")
    if existing:
        st.subheader("Optimization Results")
        rows = []
        for model_key, res in existing.items():
            rows.append(
                {
                    "model": model_key,
                    "method": res.get("method"),
                    "scoring": res.get("scoring"),
                    "cv_folds": res.get("cv_folds"),
                    "best_score": res.get("best_score"),
                    "elapsed_seconds": round(float(res.get("elapsed_seconds", 0.0)), 3),
                }
            )
        st.dataframe(rows, width="stretch")

        st.subheader("Download Optimized Model")
        downloadable = {
            str(model_key): (res or {}).get("best_estimator")
            for model_key, res in existing.items()
        }
        downloadable = {k: v for k, v in downloadable.items() if v is not None}

        if not downloadable:
            st.info("No optimized estimators available to download yet.")
        else:
            selected_key = st.selectbox(
                "Select an optimized model to download",
                options=sorted(downloadable.keys()),
                index=0,
            )
            estimator = downloadable.get(selected_key)
            try:
                payload, ext, mime = _serialize_estimator(estimator)
            except (TypeError, ValueError, AttributeError, pickle.PicklingError) as exc:
                st.error(f"Failed to serialize model: {exc}")
            else:
                fname = f"{_safe_filename(selected_key)}_optimized.{ext}"
                st.download_button(
                    "Download selected optimized model",
                    data=payload,
                    file_name=fname,
                    mime=mime,
                )

            zip_bytes, zip_err = _build_models_zip_bytes(existing)
            if zip_err is not None:
                st.info(zip_err)
            else:
                st.download_button(
                    "Download all optimized models (ZIP)",
                    data=zip_bytes,
                    file_name="optimized_models.zip",
                    mime="application/zip",
                )
