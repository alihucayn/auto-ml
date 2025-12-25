"""
Builds the final report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape as _xml_escape
import re

import numpy as np
import pandas as pd
import plotly.express as px

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PDF_DECIMALS = 4


def _finalize_report_fig(fig: Any, *, kind: str) -> Any:
    """Apply consistent layout so axis labels/ticks don't get clipped."""

    fig.update_layout(
        title_x=0.0,
        font=dict(size=12),
        xaxis=dict(automargin=True),
        yaxis=dict(automargin=True),
    )
    fig.update_xaxes(ticks="outside", ticklen=4, ticklabelposition="outside")
    fig.update_yaxes(ticks="outside", ticklen=4, ticklabelposition="outside")

    if kind in {"bar_categorical"}:
        fig.update_layout(height=520, margin=dict(l=72, r=24, t=60, b=150))
        fig.update_xaxes(tickangle=-45, tickfont=dict(size=11))
        fig.update_yaxes(tickfont=dict(size=11))
    elif kind in {"heatmap"}:

        fig.update_layout(height=640, margin=dict(l=110, r=30, t=60, b=90))
        fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
        fig.update_yaxes(tickfont=dict(size=10))
    elif kind in {"hist"}:
        fig.update_layout(height=460, margin=dict(l=72, r=24, t=60, b=90))
        fig.update_xaxes(tickfont=dict(size=11))
        fig.update_yaxes(tickfont=dict(size=11))
    else:
        fig.update_layout(height=520, margin=dict(l=72, r=24, t=60, b=90))
    return fig


def build_report_figures(
    eda_results: Any,
    *,
    max_histograms: int | None = None,
) -> list[tuple[str, str, Any]]:
    """Build Plotly figures used by the report.

    Returns a list of (key, title, figure) tuples.
    """

    figs: list[tuple[str, str, Any]] = []

    eda = eda_results or {}
    if not isinstance(eda, dict):
        return figs

    missing_tbl = _maybe_df((eda.get("missing_values") or {}).get("table"))
    if missing_tbl is not None and not missing_tbl.empty:
        view = missing_tbl.head(20).copy()
        if "missing_percent" in view.columns:
            view["missing_percent"] = pd.to_numeric(
                view["missing_percent"], errors="coerce")
        fig = px.bar(view, x="column", y="missing_percent",
                     title="Missingness by Column (%)")
        fig = _finalize_report_fig(fig, kind="bar_categorical")
        figs.append(("missingness", "Missingness by Column (%)", fig))

    out_tbl = _maybe_df((eda.get("outliers") or {}).get("table"))
    if out_tbl is not None and not out_tbl.empty:
        view = out_tbl.head(20).copy()
        if "outlier_percent" in view.columns:
            view["outlier_percent"] = pd.to_numeric(
                view["outlier_percent"], errors="coerce")
        fig = px.bar(view, x="column", y="outlier_percent",
                     title="Outlier Rate by Column (%)")
        fig = _finalize_report_fig(fig, kind="bar_categorical")
        figs.append(("outliers", "Outlier Rate by Column (%)", fig))

    corr = (eda.get("correlations") or {}).get("matrix")
    if isinstance(corr, pd.DataFrame) and not corr.empty and corr.shape[0] >= 2:
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
        fig = _finalize_report_fig(fig, kind="heatmap")
        figs.append(("correlations", "Correlation Heatmap", fig))

    dist = eda.get("distributions")
    if isinstance(dist, dict):
        results = dist.get("results") or {}
        cols = dist.get("columns") or []
        cat_cols = dist.get("categorical_columns") or []
        if isinstance(results, dict):
            ordered_cols: list[str] = []
            if isinstance(cols, list):
                ordered_cols.extend([str(c)
                                    for c in cols if str(c) in results])
            if isinstance(cat_cols, list):
                ordered_cols.extend([str(c)
                                    for c in cat_cols if str(c) in results])

            if not ordered_cols:
                ordered_cols = [str(c) for c in results.keys()]

            if max_histograms is not None:
                ordered_cols = ordered_cols[: max(0, int(max_histograms))]

            for col in ordered_cols:
                item = results.get(col) or {}
                bins = item.get("bins") or []
                counts = item.get("counts") or []
                value_counts = item.get("value_counts") or None

                if value_counts is not None:
                    vc_df = pd.DataFrame(
                        value_counts, columns=["value", "count"])
                    vc_df["value"] = vc_df["value"].astype(str)
                    title = f"Bar Chart: {col}"
                    fig = px.bar(vc_df, x="value", y="count", title=title)
                    fig = _finalize_report_fig(fig, kind="bar_categorical")
                    figs.append((f"bar_{col}", title, fig))
                elif len(bins) >= 2 and len(counts) >= 1:
                    edges = np.asarray(bins, dtype=float)
                    centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()
                    df_hist = pd.DataFrame(
                        {"bin_center": centers, "count": counts})
                    title = f"Histogram: {col}"
                    fig = px.bar(df_hist, x="bin_center",
                                 y="count", title=title)
                    fig = _finalize_report_fig(fig, kind="hist")
                    figs.append((f"hist_{col}", title, fig))

    return figs


def _fig_to_png(fig: Any) -> bytes | None:
    try:
        return fig.to_image(format="png", scale=2)
    except (ValueError, RuntimeError, TypeError, ImportError, ModuleNotFoundError):
        return None


def build_report_markdown_bundle(
    eda_results: Any,
    model_comparison: Any,
    *,
    images_subdir: str = "images",
) -> tuple[str, dict[str, bytes]]:
    """Return markdown report + image assets.

    The markdown will reference images like `images/<name>.png`.
    The returned dict maps relative paths (e.g. `images/foo.png`) to PNG bytes.
    """

    md = build_report(eda_results, model_comparison)
    assets: dict[str, bytes] = {}

    image_lines: list[str] = []
    for key, title, fig in build_report_figures(eda_results, max_histograms=None):
        png = _fig_to_png(fig)
        if png is None:
            continue

        if key == "missingness":
            filename = "missingness.png"
        elif key == "outliers":
            filename = "outliers.png"
        elif key == "correlations":
            filename = "correlations.png"
        elif key.startswith("hist_"):
            safe = key[len("hist_"):].replace(
                "/", "_").replace("\\", "_").replace(" ", "_")
            filename = f"histogram_{safe}.png"
        else:
            filename = f"{key}.png"

        rel_path = f"{images_subdir}/{filename}"
        assets[rel_path] = png
        image_lines.append(f"![{_escape_md(title)}]({rel_path})")

    if image_lines:
        md = (md.rstrip() + "\n\n" + _section("Figures",
              "\n\n".join(image_lines))).strip() + "\n"

    return md, assets


def _escape_md(text: Any) -> str:
    s = "" if text is None else str(text)
    return s.replace("|", "\\|").replace("\n", " ")


def _escape_pdf(text: Any) -> str:
    """Escape text for ReportLab Paragraph XML."""

    s = "" if text is None else str(text)
    s = s.replace("\n", " ")
    return _xml_escape(s)


def _format_number_pdf(value: Any, *, decimals: int = PDF_DECIMALS) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return f"{float(value):.{int(decimals)}f}"
    return str(value)


def _round_df_for_pdf(df: pd.DataFrame, *, decimals: int = PDF_DECIMALS) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(int(decimals))
    return out


def _df_to_markdown_table(df: pd.DataFrame, *, max_rows: int = 25) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table without extra deps."""

    if df is None or df.empty:
        return "(no data)"

    view = _round_df_for_pdf(df)
    view = view.rename(columns=lambda c: str(c).replace("_", " ").title())
    if len(view) > max_rows:
        view = view.head(max_rows)

    cols = [str(c) for c in view.columns.tolist()]
    header = "| " + " | ".join(_escape_md(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in view.iterrows():
        rows.append(
            "| "
            + " | ".join(_escape_md(_format_number_pdf(row[c])) for c in cols)
            + " |"
        )

    out = [header, sep, *rows]
    if len(df) > max_rows:
        out.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(out)


def _maybe_df(obj: Any) -> pd.DataFrame | None:
    if isinstance(obj, pd.DataFrame):
        return obj
    return None


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def build_report(eda_results, model_comparison) -> str:
    """
    Builds a markdown/HTML report.

    Args:
        eda_results: The EDA results.
        model_comparison: The model comparison results.

    Returns:
        The generated report as a string.
    """

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts: list[str] = []
    parts.append(f"# AutoML Report\n\n_Created: {created_at}_\n")

    eda = eda_results or {}
    warnings = eda.get("warnings", []) if isinstance(eda, dict) else []

    if warnings:
        warn_lines = "\n".join(f"- {_escape_md(w)}" for w in warnings)
        parts.append(_section("EDA Warnings", warn_lines))

    missing_tbl = None
    if isinstance(eda, dict):
        missing_tbl = _maybe_df((eda.get("missing_values") or {}).get("table"))

    if missing_tbl is not None:
        parts.append(_section("Missing Values",
                     _df_to_markdown_table(missing_tbl, max_rows=25)))

    outlier_tbl = None
    if isinstance(eda, dict):
        outlier_tbl = _maybe_df((eda.get("outliers") or {}).get("table"))

    if outlier_tbl is not None:
        parts.append(_section("Outliers (Summary)",
                     _df_to_markdown_table(outlier_tbl, max_rows=25)))

    corr = None
    if isinstance(eda, dict):
        corr = (eda.get("correlations") or {}).get("matrix")
    if corr is not None:
        try:
            shape = getattr(corr, "shape", None)
            parts.append(_section(
                "Correlations", f"Correlation matrix computed. Shape: {_escape_md(shape)}"))
        except (AttributeError, TypeError, ValueError):
            parts.append(
                _section("Correlations", "Correlation matrix computed."))

    if (missing_tbl is None) and (outlier_tbl is None) and (corr is None):
        parts.append(_section("EDA", "No EDA results available."))

    dist = eda.get("distributions") if isinstance(eda, dict) else None
    if isinstance(dist, dict):
        selected = dist.get("columns") or []
        results = dist.get("results") or {}
        if selected and isinstance(results, dict):
            shown = []
            for col in selected[:5]:
                stats = (results.get(col) or {}).get("stats") or {}
                mean = stats.get("mean")
                std = stats.get("std")
                shown.append(
                    f"- {col}: mean={_format_number_pdf(mean)}, std={_format_number_pdf(std)}"
                )
            parts.append(_section("Distributions (Summary)",
                         "\n".join(shown) if shown else "(no data)"))

    comparison = model_comparison or {}
    comparison_table = None
    rank_metric = None
    if isinstance(comparison, dict):
        rank_metric = comparison.get("rank_metric")
        comparison_table = _maybe_df(comparison.get("table"))
    elif isinstance(comparison, pd.DataFrame):
        comparison_table = comparison

    if comparison_table is not None and not comparison_table.empty:
        summary_lines = []
        if rank_metric:
            summary_lines.append(f"Rank metric: **{_escape_md(rank_metric)}**")

        best_model = None
        if "rank" in comparison_table.columns:
            try:
                best_row = comparison_table.sort_values("rank").iloc[0]
                best_model = best_row.get("model")
            except (KeyError, IndexError, TypeError, ValueError):
                best_model = None
        if best_model is None and "model" in comparison_table.columns:
            best_model = comparison_table.iloc[0].get("model")

        if best_model is not None:
            summary_lines.append(f"Best model: **{_escape_md(best_model)}**")

        summary = "\n\n".join(summary_lines) if summary_lines else ""
        table_md = _df_to_markdown_table(comparison_table, max_rows=25)
        body = (summary + "\n\n" + table_md).strip()
        parts.append(_section("Model Comparison", body))
    else:
        parts.append(_section("Model Comparison",
                     "No model comparison results available."))

    return "\n\n".join(p.strip() for p in parts if p.strip()) + "\n"


def build_report_pdf(eda_results, model_comparison) -> bytes:
    """Build a PDF version of the report.

    Uses `reportlab` (pure Python) to avoid system-level dependencies.
    The PDF rendering is intentionally simple: headings, bullets, paragraphs,
    and Markdown tables shown in a monospace block.
    """

    md = build_report(eda_results, model_comparison)

    styles = getSampleStyleSheet()
    body = styles["BodyText"]

    h1 = ParagraphStyle(
        "ReportHeading1",
        parent=styles["Heading1"],
        keepWithNext=True,
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "ReportHeading2",
        parent=styles["Heading2"],
        keepWithNext=True,
        spaceAfter=8,
    )
    h3 = ParagraphStyle(
        "ReportHeading3",
        parent=styles["Heading3"],
        keepWithNext=True,
        spaceAfter=6,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=body,
        fontSize=8,
        leading=10,
        spaceAfter=0,
        spaceBefore=0,
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=table_cell,
        fontSize=8,
        leading=10,
    )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="AutoML Report",
    )

    story: list[Any] = []
    in_table_block = False
    table_lines: list[str] = []
    last_was_heading = False

    in_code_fence = False
    code_fence_lines: list[str] = []

    def _md_inline_to_rl(text: Any) -> str:
        """Convert a small Markdown subset into ReportLab Paragraph XML."""

        s = "" if text is None else str(text)
        s = s.replace("\n", " ").strip()
        if not s:
            return ""

        code_spans: list[str] = []

        def _stash_code(m: re.Match[str]) -> str:
            code_spans.append(m.group(1))
            return f"\uFFF0CODE{len(code_spans) - 1}\uFFF1"

        s = re.sub(r"`([^`]+)`", _stash_code, s)

        def _link(m: re.Match[str]) -> str:
            label = _xml_escape(m.group(1).strip())
            url = _xml_escape(m.group(2).strip())
            return f'<a href="{url}">{label}</a>'

        s = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", _link, s)

        s = _xml_escape(s)

        for idx, code in enumerate(code_spans):
            token = _xml_escape(f"\uFFF0CODE{idx}\uFFF1")
            code_escaped = _xml_escape(code)
            s = s.replace(token, f'<font face="Courier">{code_escaped}</font>')

        s = re.sub(r"\*\*([^*]+?)\*\*", r"<b>\1</b>", s)

        s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", s)

        s = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", s)

        return s

    png_by_key: dict[str, bytes] = {}
    hist_pngs: list[bytes] = []
    bar_pngs: list[bytes] = []
    for key, _title, fig in build_report_figures(eda_results, max_histograms=None):
        png = _fig_to_png(fig)
        if png is None:
            continue
        if key.startswith("hist_"):
            hist_pngs.append(png)
        elif key.startswith("bar_"):
            bar_pngs.append(png)
        else:
            png_by_key[key] = png

    missing_img = png_by_key.get("missingness")
    outliers_img = png_by_key.get("outliers")
    corr_img = png_by_key.get("correlations")
    dist_imgs = hist_pngs + bar_pngs

    current_section: str | None = None
    corr_image_emitted = False
    dist_images_emitted = False

    def _add_image(png: bytes, *, max_height_inches: float = 3.2) -> None:
        """Embed a PNG scaled to fit width while preserving aspect ratio."""

        reader = ImageReader(BytesIO(png))
        img_w, img_h = reader.getSize()
        if not img_w or not img_h:
            return

        target_w = float(doc.width)
        scale = target_w / float(img_w)
        target_h = float(img_h) * scale

        max_h = float(max_height_inches) * inch
        if target_h > max_h:
            scale = max_h / float(img_h)
            target_h = max_h
            target_w = float(img_w) * scale

        story.append(Image(BytesIO(png), width=target_w, height=target_h))
        story.append(Spacer(1, 10))

    def _parse_markdown_table(lines: list[str]) -> list[list[str]]:
        rows: list[list[str]] = []
        for ln in lines:
            stripped_ln = ln.strip()
            if not (stripped_ln.startswith("|") and stripped_ln.endswith("|")):
                continue
            cells = [c.strip() for c in stripped_ln.strip("|").split("|")]
            rows.append(cells)

        if len(rows) >= 2:
            sep = rows[1]
            is_sep = True
            for c in sep:
                c2 = c.replace(":", "").strip()
                if len(c2) < 3 or any(ch != "-" for ch in c2):
                    is_sep = False
                    break
            if is_sep:
                rows.pop(1)

        max_cols = max((len(r) for r in rows), default=0)
        norm: list[list[str]] = []
        for r in rows:
            if len(r) < max_cols:
                r = r + [""] * (max_cols - len(r))
            norm.append(r)
        return norm

    def _table_from_md(lines: list[str]):
        data = _parse_markdown_table(lines)
        if not data:
            return None

        header_row = [Paragraph(_escape_pdf(c), table_header) for c in data[0]]
        body_rows = [[Paragraph(_escape_pdf(c), table_cell)
                      for c in row] for row in data[1:]]
        table_data = [header_row, *body_rows]

        available_width = float(doc.width)
        col_count = len(table_data[0])

        max_lens: list[int] = [1] * col_count
        for col_idx in range(col_count):
            best = 1
            for row in data:
                if col_idx < len(row):
                    best = max(
                        best, len(str(row[col_idx])) if row[col_idx] is not None else 0)
            max_lens[col_idx] = min(best, 40)

        total = float(sum(max_lens)) if sum(max_lens) else 1.0
        min_w = 0.8 * inch
        col_widths = [(ml / total) * available_width for ml in max_lens]

        col_widths = [max(min_w, w) for w in col_widths]
        width_sum = float(sum(col_widths))
        if width_sum > available_width:
            scale = available_width / width_sum
            col_widths = [w * scale for w in col_widths]

        tbl = Table(
            table_data,
            colWidths=col_widths,
            repeatRows=1,
            hAlign="LEFT",
        )
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        return tbl

    for raw_line in md.splitlines():
        line = raw_line.rstrip("\n")

        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code_fence:
                story.append(Preformatted(
                    "\n".join(code_fence_lines), styles["Code"]))
                story.append(Spacer(1, 10))
                code_fence_lines = []
                in_code_fence = False
            else:
                in_code_fence = True
                code_fence_lines = []
            continue

        if in_code_fence:
            code_fence_lines.append(line)
            continue

        is_table_line = ("|" in line) and line.strip().startswith(
            "|") and line.strip().endswith("|")

        if is_table_line:
            in_table_block = True
            table_lines.append(line)
            continue

        if in_table_block and (not is_table_line):

            tbl = _table_from_md(table_lines)
            if tbl is not None:
                story.append(tbl)
            else:
                story.append(Preformatted(
                    "\n".join(table_lines), styles["Code"]))
            story.append(Spacer(1, 10))

            if current_section == "Missing Values" and missing_img is not None:
                _add_image(missing_img, max_height_inches=3.0)
            if current_section == "Outliers (Summary)" and outliers_img is not None:
                _add_image(outliers_img, max_height_inches=3.0)

            table_lines = []
            in_table_block = False

        if not stripped:

            if not last_was_heading:
                story.append(Spacer(1, 8))
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(_md_inline_to_rl(stripped[2:]), h1))
            last_was_heading = True
            continue

        if stripped.startswith("## "):
            current_section = stripped[3:].strip()
            story.append(Paragraph(_md_inline_to_rl(current_section), h2))
            last_was_heading = True

            if current_section == "Correlations" and (corr_img is not None) and (not corr_image_emitted):
                _add_image(corr_img, max_height_inches=4.2)
                corr_image_emitted = True

            if current_section == "Distributions (Summary)" and dist_imgs and (not dist_images_emitted):
                for img in dist_imgs:
                    _add_image(img, max_height_inches=3.0)
                dist_images_emitted = True
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(_md_inline_to_rl(stripped[4:]), h3))
            last_was_heading = True
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):

            story.append(
                Paragraph(f"• {_md_inline_to_rl(stripped[2:])}", body))
            last_was_heading = False
            continue

        story.append(Paragraph(_md_inline_to_rl(stripped), body))
        last_was_heading = False

    if in_table_block and table_lines:
        tbl = _table_from_md(table_lines)
        if tbl is not None:
            story.append(tbl)
        else:
            story.append(Preformatted("\n".join(table_lines), styles["Code"]))

        if current_section == "Missing Values" and missing_img is not None:
            _add_image(missing_img, max_height_inches=3.0)
        if current_section == "Outliers (Summary)" and outliers_img is not None:
            _add_image(outliers_img, max_height_inches=3.0)

    if in_code_fence and code_fence_lines:
        story.append(Preformatted("\n".join(code_fence_lines), styles["Code"]))

    doc.build(story)
    return buf.getvalue()
