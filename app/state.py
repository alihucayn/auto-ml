"""
Manages the session state for the Streamlit application.
"""

import streamlit as st


def initialize_session():
    """Initializes the session state with default values."""
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'eda_results' not in st.session_state:
        st.session_state.eda_results = None
    if 'issues_results' not in st.session_state:
        st.session_state.issues_results = None
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'model_comparison' not in st.session_state:
        st.session_state.model_comparison = None
    if 'report' not in st.session_state:
        st.session_state.report = None


def reset_session():
    """Resets the session state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session()
