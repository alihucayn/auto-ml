"""
Streamlit entry point for the AutoML system.

This script orchestrates the UI and the overall workflow of the application.
"""

from __future__ import annotations
from ui import (
    comparison_view,
    eda_view,
    inference_view,
    optimization_view,
    preprocessing_view,
    report_view,
    training_view,
    upload,
)
from app import state

import sys
from pathlib import Path

import streamlit as st


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def apply_dark_emerald_theme() -> None:
    """Inject a dark emerald theme for a research-grade look."""

    st.markdown(
        """
        <style>
        :root {
            --bg-main: #0B0F0E;
            --bg-sidebar: #0E1513;
            --bg-card: #111C18;
            --accent: #1DB954;
            --accent-2: #157347;
            --text: #E6ECEA;
            --text-muted: #A8B3AF;
            --border: #1F2A27;
            --hover: rgba(29,185,84,0.12);
            --radius: 10px;
            --font: "Inter", "DM Sans", "Segoe UI", sans-serif;
            /* Align Streamlit theme tokens so native widgets inherit emerald */
            --primary-color: #1DB954;
            --primary: #1DB954;
            --secondary: #157347;
            --success: #1DB954;
            --warning: #f2c94c;
        }

        body, .main, .stApp {
            background-color: var(--bg-main) !important;
            background-image:
                radial-gradient(circle at 20% 20%, rgba(29,185,84,0.06) 0, transparent 28%),
                radial-gradient(circle at 80% 10%, rgba(21,115,71,0.05) 0, transparent 24%),
                radial-gradient(circle at 30% 80%, rgba(29,185,84,0.04) 0, transparent 26%),
                linear-gradient(135deg, rgba(17,28,24,0.9) 0%, rgba(11,15,14,0.95) 40%, rgba(11,15,14,0.98) 100%);
            color: var(--text) !important;
            font-family: var(--font);
        }

        /* Sidebar compact, secondary */
        section[data-testid="stSidebar"], .stSidebar {
            background: var(--bg-sidebar) !important;
            color: var(--text) !important;
            border-right: 1px solid var(--border);
            font-size: 0.95rem;
        }
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            color: var(--accent) !important;
            margin-bottom: 0.4rem;
        }

        /* Card-like containers */
        .block-container { padding-top: 1.2rem; padding-bottom: 1.6rem; }
        .stDataFrame, .stTable, .stMetric, .stAlert, .st-expander, .stDataEditor, .stTabs > div {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }

        /* Headings & text */
        h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: var(--accent) !important;
            letter-spacing: 0.01em;
            font-weight: 600;
            text-shadow: 0 0 10px rgba(29,185,84,0.16);
        }
        p, label, span, div { color: var(--text) !important; text-shadow: 0 0 6px rgba(29,185,84,0.06); }
        .stMarkdown small, .stCaption, .stMarkdown p em { color: var(--text-muted) !important; }

        /* Buttons */
        .stButton button, .stDownloadButton button {
            background: var(--accent-2) !important;
            color: var(--text) !important;
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
            box-shadow: none !important;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            background: var(--accent) !important;
            color: var(--bg-main) !important;
        }

        /* Inputs & selects */
        input, textarea, select, .stTextInput > div > input, .stNumberInput input, .stSelectbox > div > div, .stDateInput input {
            background: var(--bg-card) !important;
            color: var(--text) !important;
            border-radius: var(--radius) !important;
            border: 1px solid var(--border) !important;
        }
        .stMultiSelect > div, .stSelectbox > div {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
        }
        .stMultiSelect [data-baseweb="tag"] {
            background: var(--hover) !important;
            color: var(--text) !important;
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
        }
        .stSelectbox [role="listbox"], .stMultiSelect [role="listbox"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
        }
        .stSelectbox [role="option"]:hover, .stMultiSelect [role="option"]:hover {
            background: var(--hover) !important;
        }
        /* Backstop slider handles/tracks to emerald if class names differ */
        [role="slider"] { background: var(--accent) !important; box-shadow: 0 4px 14px rgba(29,185,84,0.35) !important; }
        [role="slider"]:focus-visible { outline: 2px solid var(--accent) !important; }
        .stSlider div[data-baseweb="slider"] [data-baseweb="track"] { background: #1a241f !important; }
        .stSlider div[data-baseweb="slider"] [data-baseweb="track"] > div {
            background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%) !important;
        }
        .stSlider div[data-baseweb="slider"] [data-baseweb="thumb"] { background: var(--accent) !important; }
        .stSlider div[data-baseweb="slider"] [data-baseweb="thumb"]:focus { outline: 2px solid var(--accent) !important; }
        .stSlider div[data-baseweb="slider"] [data-baseweb="thumb"]::after { background: var(--accent) !important; }

        /* Sliders */
        .stSlider > div[data-baseweb="slider"] > div {
            background: #1a241f !important;
            height: 8px;
            border-radius: 999px;
        }
        .stSlider [data-testid="stSliderThumb"] {
            width: 18px; height: 18px;
            background: var(--accent) !important;
            box-shadow: 0 4px 14px rgba(29,185,84,0.35) !important;
            border: 2px solid var(--bg-main) !important;
        }
        .stSlider .css-1h1j0y3 {
            background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%) !important;
            height: 8px;
            border-radius: 999px;
        }
        .stSlider label { color: var(--text) !important; }
        .stSlider .css-1pahdxg, .stSlider .css-1h1j0y3 + div { color: var(--text-muted) !important; }
        .stSelectbox [role="option"]:focus, .stMultiSelect [role="option"]:focus { background: var(--hover) !important; }
        .stSelectbox div[data-baseweb="select"] div:focus, .stMultiSelect div[data-baseweb="select"] div:focus {
            outline: 1px solid var(--accent) !important;
        }

        /* Checkbox / radio */
        .stCheckbox input[type='checkbox'], .stRadio input[type='radio'] { accent-color: var(--accent); }
        .stRadio label, .stCheckbox label { color: var(--text-muted) !important; }
        .stRadio [role="radio"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            padding: 6px 8px !important;
        }
        .stRadio [aria-checked="true"] {
            background: var(--hover) !important;
            border-color: var(--accent) !important;
            color: var(--accent) !important;
        }
        .stRadio [aria-checked="true"] svg { color: var(--accent) !important; fill: var(--accent) !important; }

        /* Tables */
        .stDataFrame tbody tr { background: var(--bg-card) !important; }
        .stDataFrame tbody tr:nth-child(odd) { background: #0d1613 !important; }
        .stDataFrame thead th { color: var(--accent) !important; border-bottom: 1px solid var(--border) !important; }
        .stDataFrame td, .stDataFrame th { border: 1px solid var(--border) !important; }

        /* Alerts */
        .stAlert {
            border-radius: var(--radius) !important;
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
        }

        /* Expanders */
        .st-expander {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
        }
        .st-expander p { color: var(--text) !important; }

        /* Tabs */
        .stTabs [data-baseweb="tab"] button { color: var(--text-muted) !important; }
        .stTabs [aria-selected="true"] button { color: var(--text) !important; border-bottom: 2px solid var(--accent); }

        /* Hover highlights */
        .stButton button:hover, .stDownloadButton button:hover, .stSelectbox [role="option"]:hover, .stMultiSelect [role="option"]:hover {
            filter: brightness(1.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home():
    """Landing page for the AutoML system."""

    st.markdown(
        """
        <div style="text-align:center; margin-top: 0.5rem;">
            <h1 style="margin-bottom: 0.2rem;">Automated Machine Learning Pipeline</h1>
            <h3 style="margin-top: 0; color: var(--text-muted);">Machine Learning (CS-245) — Project</h3>
            <p style="max-width: 820px; margin: 0.6rem auto 1.2rem; font-size: 1.05rem;">
                An end-to-end AutoML system that automates data analysis, preprocessing, model training, optimization, comparison, and reporting using Streamlit and scikit-learn.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            """
            **Project Overview** — This project streamlines the full ML lifecycle so beginners and advanced users can focus on insights instead of boilerplate. It covers data upload with schema hints, dataset issue detection (missingness, imbalance, high cardinality, datatype mismatches), exploratory data analysis, preprocessing (imputation, encoding, scaling, outlier handling), model training and optimization, side-by-side comparisons, inference, and report generation—built with Streamlit for the interface and scikit-learn for the pipelines.
            """
        )

    st.write("")

    # Features
    st.markdown("### System Capabilities")
    feat_cols = st.columns(2)
    features_left = [
        "📥 Guided dataset upload with schema and target selection",
        "⚠️ Dataset issue detection (missingness, imbalance, high cardinality, datatype mismatches)",
        "📊 Automated Exploratory Data Analysis with visuals",
        "🧹 Preprocessing pipeline: imputation, encoding, scaling, outlier handling",
    ]
    features_right = [
        "🤖 Multiple ML models (Logistic Regression, KNN, SVM, Decision Tree, Naive Bayes)",
        "🔍 Hyperparameter optimization (Grid & Random Search)",
        "📈 Model evaluation, ranking, and comparison views",
        "📄 Report generation and inference on new data",
    ]
    with feat_cols[0]:
        st.markdown("\n".join([f"- {item}" for item in features_left]))
    with feat_cols[1]:
        st.markdown("\n".join([f"- {item}" for item in features_right]))

    st.write("")

    # Team
    st.markdown("### Team Contributions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            🧠 **Ali Hussain (477193) — System Architecture & ML Pipeline**
            - Backend architecture and modular design
            - Automated EDA logic
            - Dataset issue detection
            - Preprocessing pipeline design
            - ML model training workflows
            - Hyperparameter optimization
            - Evaluation metrics & reproducibility
            - Focus on scalability and maintainability
            """
        )
    with col2:
        st.markdown(
            """
            🎨 **Labib Kamran (467183) — UI, Visualization & Reporting**
            - Streamlit UI flow design
            - Dataset preview and EDA visualizations
            - User-driven preprocessing options
            - Model comparison tables and charts
            - Report generation (Markdown / PDF)
            - Usability and error handling
            """
        )

    st.write("")

    # Workflow
    st.markdown("### Workflow Overview")
    st.markdown(
        "<div style='font-size:1.05rem; font-weight:500;'>"
        "Upload Dataset → Issue Detection → EDA → Preprocessing → Training → Optimization → Comparison → Report → Inference"
        "</div>",
        unsafe_allow_html=True,
    )

    st.write("")

    # CTA
    cta_col = st.container()
    with cta_col:
        if st.button("Start AutoML Pipeline", type="primary"):
            st.session_state["nav_page"] = "Upload"
            st.experimental_rerun()


def styled_container():
    """Convenience wrapper for a card-like container."""

    return st.container()


def section_header(title: str):
    """Render a consistent section header."""

    st.markdown(f"## {title}")


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        layout="wide",
        page_title="AutoML System",
        initial_sidebar_state="expanded",
    )

    apply_dark_emerald_theme()

    state.initialize_session()

    nav_options = [
        "Home",
        "Upload",
        "Issue Detection",
        "EDA",
        "Preprocessing",
        "Training",
        "Optimization",
        "Comparison",
        "Report",
        "Inference",
    ]

    default_page = st.session_state.get("nav_page", "Home")
    if default_page not in nav_options:
        default_page = "Home"

    st.title("AutoML System")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        nav_options,
        index=nav_options.index(default_page),
        key="nav_page",
    )

    if page == "Home":
        render_home()
    elif page == "Upload":
        upload.show()
    elif page == "Issue Detection":
        eda_view.show_issues()
    elif page == "EDA":
        eda_view.show()
    elif page == "Preprocessing":
        preprocessing_view.show()
    elif page == "Training":
        training_view.show()
    elif page == "Optimization":
        optimization_view.show()
    elif page == "Comparison":
        comparison_view.show()
    elif page == "Report":
        report_view.show()
    elif page == "Inference":
        inference_view.show()


if __name__ == "__main__":
    main()
