"""
UI component for displaying the final report.
"""

from __future__ import annotations

from io import BytesIO
import zipfile

import streamlit as st
from reports import report_builder


def show():
    """Shows the report page."""
    st.header("8. Report")

    if not st.session_state.get("eda_results") or not st.session_state.get("model_comparison"):
        st.warning("Please run EDA and model comparison first.")
        return

    col_action, _ = st.columns([1, 3])
    with col_action:
        regenerate = st.button("Regenerate Report")

    cache_key = "_report_render_cache_v1"
    dataset_sig = st.session_state.get("dataset_signature")
    target_col = st.session_state.get("target_column")
    desired_sig = (dataset_sig, target_col)

    ai_md = None

    cache = st.session_state.get(cache_key) or {}
    need_build = regenerate or (cache.get("sig") != desired_sig)

    if need_build:
        with st.spinner("Building report (tables, images, downloads)…"):
            report_text = report_builder.build_report(
                st.session_state.eda_results,
                st.session_state.model_comparison,
            )

            bundle_md, bundle_assets = report_builder.build_report_markdown_bundle(
                st.session_state.eda_results,
                st.session_state.model_comparison,
                images_subdir="images",
            )

            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                root = "automl_report/"
                zf.writestr(root + "report.md", bundle_md.encode("utf-8"))
                for rel_path, data in bundle_assets.items():
                    zf.writestr(root + rel_path, data)

            pdf_bytes = None
            pdf_error = None
            try:
                pdf_bytes = report_builder.build_report_pdf(
                    st.session_state.eda_results,
                    st.session_state.model_comparison,
                )
            except (ImportError, ModuleNotFoundError, RuntimeError, ValueError, TypeError, OSError) as exc:
                pdf_error = f"{type(exc).__name__}: {exc}"

            figs = report_builder.build_report_figures(
                st.session_state.eda_results,
                max_histograms=None,
            )

            cache = {
                "sig": desired_sig,
                "report_text": report_text,
                "zip_bytes": zip_buf.getvalue(),
                "zip_has_images": bool(bundle_assets),
                "pdf_bytes": pdf_bytes,
                "pdf_error": pdf_error,
                "figs": figs,
            }
            st.session_state[cache_key] = cache

    report_text = cache.get("report_text") or ""
    st.markdown(report_text, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if not cache.get("zip_has_images"):
            st.info(
                "Markdown images could not be exported. Ensure `kaleido` is installed for Plotly image export."
            )

        st.download_button(
            label="Download Markdown + Images (ZIP)",
            data=cache.get("zip_bytes") or b"",
            file_name="automl_report_bundle.zip",
            mime="application/zip",
            width="stretch",
        )

    with col2:
        if cache.get("pdf_error"):
            st.error("Failed to build PDF report.")
            st.code(cache.get("pdf_error"), language="text")

        pdf_bytes = cache.get("pdf_bytes")
        if pdf_bytes:
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name="automl_report.pdf",
                mime="application/pdf",
                width="stretch",
            )

    st.divider()
    st.subheader("Report Visualizations")
    figs = cache.get("figs") or []
    if not figs:
        st.info("No visualizations available for the current EDA results.")
    else:
        for _key, title, fig in figs:
            st.markdown(f"**{title}**")
            st.plotly_chart(fig, width="stretch")
