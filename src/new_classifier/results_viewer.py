"""Streamlit viewer for agglomerative k8 classification results."""

import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from src.new_classifier.experiment_manager import (
    EXPERIMENT_RESULTS_NAME,
    list_experiment_dirs,
    load_experiment_metadata,
)
from src.utils.constants import DATA_DIR, EXPERIMENTS_DIR, PROJECT_ROOT

LEGACY_RESULTS_FILE = DATA_DIR / "categories_agglomerative_k8_classification_results.csv"
ORIGIN_LABELS_FILE = DATA_DIR / "results.csv"
DEFAULT_CATEGORIES_FILE = PROJECT_ROOT / "src" / "new_classifier" / "categories_agglomerative_k8.csv"
NOT_RELEVANT = "לא רלוונטי"
ACCENT = "#2563eb"

CHART_COLORS = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#64748b",
]

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.hebrew {
    font-family: 'Heebo', 'Arial Hebrew', 'David', sans-serif;
    direction: rtl;
    text-align: right;
    unicode-bidi: plaintext;
}

.stApp {
    background: linear-gradient(160deg, #f7f9fc 0%, #eef2f7 45%, #f4f6fa 100%);
}

.main-header h1 {
    font-weight: 700;
    color: #1a2b4a;
    margin-bottom: 0.25rem;
    font-size: 2rem;
}

.main-header p {
    color: #5a6b82;
    font-size: 1.05rem;
    margin: 0;
}

.sentence-card {
    background: white;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.85rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    direction: rtl;
    text-align: right;
}

.sentence-body {
    line-height: 1.8;
    font-size: 1.1rem;
    font-weight: 500;
    color: #0f172a;
}

.card-divider {
    border-top: 1px solid #e2e8f0;
    margin: 1rem 0 0.65rem;
}

.origin-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #94a3b8;
    margin-bottom: 0.35rem;
    direction: ltr;
    text-align: left;
}

.origin-tag {
    font-size: 0.88rem;
    font-weight: 400;
    color: #475569;
    background: #f8fafc;
    border-radius: 6px;
    padding: 0.45rem 0.65rem;
    line-height: 1.55;
    border: 1px solid #e2e8f0;
}

.panel-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 0.75rem;
}

.nav-hint {
    text-align: center;
    color: #5a6b82;
    font-size: 0.9rem;
    margin: 0.25rem 0;
}

.category-header {
    direction: rtl;
    text-align: right;
    font-size: 1rem;
    margin-bottom: 0.75rem;
    line-height: 1.6;
}

.category-header .cat-name {
    font-weight: 600;
    color: #2563eb;
}

.category-header .count-he {
    font-weight: 400;
    color: #64748b;
    font-size: 0.9rem;
}

.cat-row-label {
    direction: rtl;
    text-align: right;
    font-family: 'Heebo', 'Arial Hebrew', sans-serif;
    font-size: 0.88rem;
    line-height: 1.45;
    border-radius: 8px;
    padding: 0.55rem 0.7rem;
    min-height: 2.5rem;
    border: 1px solid #e2e8f0;
    background: white;
}

.cat-row-label.active {
    border-color: #2563eb;
    background: #eff6ff;
    color: #1e40af;
    font-weight: 600;
}

.prompt-block {
    direction: rtl;
    text-align: right;
    font-family: 'Heebo', 'Arial Hebrew', sans-serif;
    white-space: pre-wrap;
    word-wrap: break-word;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.15rem;
    line-height: 1.65;
    font-size: 0.92rem;
    color: #1e293b;
    margin: 0;
}

.prompt-placeholder {
    background: #fef9c3;
    color: #854d0e;
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
    direction: ltr;
    unicode-bidi: embed;
    font-weight: 600;
}

.exp-meta {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 1rem;
}
</style>
"""


def _normalize_sentence(text: str) -> str:
    return str(text).strip().replace("  ", " ")


def _resolve_categories_file(metadata: Optional[Dict[str, Any]]) -> Path:
    if metadata and metadata.get("categories_file"):
        path = Path(metadata["categories_file"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            return path
    return DEFAULT_CATEGORIES_FILE


def _available_experiments() -> List[str]:
    return [d.name for d in list_experiment_dirs() if (d / EXPERIMENT_RESULTS_NAME).exists()]


@st.cache_data
def load_data(experiment_id: str) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    exp_dir = EXPERIMENTS_DIR / experiment_id
    results_path = exp_dir / EXPERIMENT_RESULTS_NAME
    if not results_path.exists():
        results_path = LEGACY_RESULTS_FILE

    metadata = load_experiment_metadata(exp_dir) if exp_dir.exists() else {}
    categories = pd.read_csv(_resolve_categories_file(metadata))
    results = pd.read_csv(results_path)

    if ORIGIN_LABELS_FILE.exists():
        origin_labels = pd.read_csv(ORIGIN_LABELS_FILE)
        origin_labels["sent_norm"] = origin_labels["origin_sentence"].map(_normalize_sentence)
        origin_labels = origin_labels.drop_duplicates("sent_norm", keep="first")
        label_map = origin_labels.set_index("sent_norm")["label_0"]
        results["sent_norm"] = results["origin_sentence"].map(_normalize_sentence)
        results["original_category"] = results["sent_norm"].map(label_map)
        results = results.drop(columns=["sent_norm"])
    else:
        results["original_category"] = pd.NA

    results["original_category"] = results["original_category"].fillna(results["category"])

    known_labels = set(categories["label"])
    results["new_category"] = results["new_category"].apply(
        lambda x: NOT_RELEVANT if x not in known_labels else x
    )

    ordered_labels = categories["label"].tolist() + [NOT_RELEVANT]
    return results, ordered_labels, metadata


def render_header(experiment_id: str, metadata: Dict[str, Any]):
    model_line = ""
    if metadata:
        model_line = (
            f"{metadata.get('provider', '?')} / {metadata.get('model', '?')}"
            f" · {metadata.get('date', '?')}"
        )
    subtitle = f"Experiment {experiment_id}"
    if model_line:
        subtitle += f" · {model_line}"
    subtitle += " · distribution and sentence browser"

    st.markdown(
        f"""
        <div class="main-header">
            <h1>Legal Sentence Classification Results</h1>
            <p>{html_lib.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _highlight_placeholder(text: str, placeholder: str) -> str:
    escaped = html_lib.escape(text)
    token = html_lib.escape(placeholder)
    return escaped.replace(
        token,
        f'<span class="prompt-placeholder">{token}</span>',
    )


def render_prompt_panel(metadata: Dict[str, Any]) -> None:
    if not metadata:
        st.info("No experiment metadata found for this run.")
        return

    prompt = metadata.get("prompt", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Provider", metadata.get("provider", "—"))
    c2.metric("Model", metadata.get("model", "—"))
    c3.metric("Date", metadata.get("date", "—"))
    c4.metric("Results", f"{metadata.get('result_count', '—'):,}" if metadata.get("result_count") else "—")

    st.markdown(
        f'<p class="exp-meta">Categories: {html_lib.escape(str(metadata.get("categories_file", "—")))}'
        f" · input scope: {html_lib.escape(str(metadata.get('input_scope', '—')))}"
        f" · temperature: {metadata.get('temperature', '—')}</p>",
        unsafe_allow_html=True,
    )

    system_prompt = prompt.get("system_prompt", "")
    user_prompt = prompt.get("user_prompt_template", "")
    placeholder = prompt.get("sentence_placeholder", "<SENTENCE>")

    st.markdown("#### System prompt")
    st.html(f'<pre class="prompt-block hebrew">{html_lib.escape(system_prompt)}</pre>')

    st.markdown("#### User prompt template")
    st.html(
        f'<pre class="prompt-block hebrew">{_highlight_placeholder(user_prompt, placeholder)}</pre>'
    )

    with st.expander("Raw metadata (JSON)"):
        st.code(json.dumps(metadata, ensure_ascii=False, indent=2), language="json")


def render_sentence_card(row: pd.Series):
    original = str(row["original_category"]).strip()
    sentence = str(row["origin_sentence"]).strip()

    parts = [
        '<div class="sentence-card">',
        f'<div class="sentence-body hebrew">{html_lib.escape(sentence)}</div>',
    ]
    if original and original != NOT_RELEVANT:
        parts.extend(
            [
                '<div class="card-divider"></div>',
                '<div class="origin-label">Original category</div>',
                f'<div class="origin-tag hebrew">{html_lib.escape(original)}</div>',
            ]
        )
    parts.append("</div>")

    st.html("".join(parts))


def category_label(cat: str, counts: pd.Series, total: int) -> str:
    count = int(counts.get(cat, 0))
    pct = 100 * count / total
    return f"{cat}  ·  {count} ({pct:.0f}%)"


def render_category_row(
    cat: str,
    counts: pd.Series,
    total: int,
    is_selected: bool,
    index: int,
) -> bool:
    css = "cat-row-label active" if is_selected else "cat-row-label"
    col_text, col_btn = st.columns([7, 1], gap="small", vertical_alignment="center")
    picked = False
    with col_text:
        st.html(
            f'<div class="{css}">{html_lib.escape(category_label(cat, counts, total))}</div>'
        )
    with col_btn:
        if st.button(
            "●" if is_selected else "○",
            key=f"catpick_{index}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            picked = True
    return picked


def render_category_nav(labels: List[str], current_idx: int, key_prefix: str) -> None:
    c_prev, c_mid, c_next = st.columns([1, 2, 1])
    with c_prev:
        if st.button(
            "← Prev",
            disabled=current_idx == 0,
            key=f"{key_prefix}_prev",
            use_container_width=True,
        ):
            st.session_state.selected_category = labels[current_idx - 1]
            st.session_state.page = 0
            st.rerun()
    with c_mid:
        st.markdown(
            f'<p class="nav-hint">{current_idx + 1} / {len(labels)}</p>',
            unsafe_allow_html=True,
        )
    with c_next:
        if st.button(
            "Next →",
            disabled=current_idx == len(labels) - 1,
            key=f"{key_prefix}_next",
            use_container_width=True,
        ):
            st.session_state.selected_category = labels[current_idx + 1]
            st.session_state.page = 0
            st.rerun()


def main():
    st.set_page_config(
        page_title="Classification Results",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)

    experiments = _available_experiments()
    if not experiments:
        st.error("No experiment folders found under data/experiments/.")
        st.stop()

    with st.sidebar:
        st.markdown("### Experiment")
        experiment_id = st.selectbox(
            "Select experiment",
            experiments,
            index=len(experiments) - 1,
            format_func=lambda x: x,
        )

    results, ordered_labels, metadata = load_data(experiment_id)
    counts = results["new_category"].value_counts()
    classified = results[results["new_category"] != NOT_RELEVANT]

    render_header(experiment_id, metadata)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total sentences", f"{len(results):,}")
    c2.metric("Classified", f"{len(classified):,}")
    c3.metric("Not relevant", f"{counts.get(NOT_RELEVANT, 0):,}")
    c4.metric("Categories", len(ordered_labels) - 1)

    tab_overview, tab_browse, tab_prompt = st.tabs(
        ["Distribution", "Sentences by category", "Experiment & prompt"]
    )

    with tab_overview:
        exclude_unknown = st.checkbox("Exclude not relevant from distribution", value=False)
        dist_labels = [l for l in ordered_labels if l != NOT_RELEVANT or not exclude_unknown]

        dist = (
            results["new_category"]
            .value_counts()
            .reindex(dist_labels, fill_value=0)
            .reset_index()
        )
        dist.columns = ["category", "count"]
        dist_total = int(dist["count"].sum())
        dist["pct"] = (100 * dist["count"] / dist_total).round(1) if dist_total else 0

        if exclude_unknown:
            st.caption(f"Percentages based on {dist_total:,} classified sentences (not relevant excluded).")

        color_map = {
            label: CHART_COLORS[i % len(CHART_COLORS)]
            for i, label in enumerate(dist_labels)
        }
        fig = px.bar(
            dist,
            x="count",
            y="category",
            orientation="h",
            color="category",
            color_discrete_map=color_map,
            text=dist.apply(lambda r: f"{int(r['count'])} ({r['pct']}%)", axis=1),
            category_orders={"category": list(reversed(dist_labels))},
        )
        fig.update_layout(
            height=520,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            font=dict(size=13),
            yaxis=dict(tickfont=dict(family="Heebo, Arial Hebrew, sans-serif", size=12)),
            xaxis_title="Number of sentences",
            yaxis_title="",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

        table = dist.sort_values("count", ascending=False).rename(
            columns={"category": "Category", "count": "Count", "pct": "%"}
        )
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn("Category", width="large"),
                "Count": st.column_config.NumberColumn("Count"),
                "%": st.column_config.NumberColumn("%", format="%.1f"),
            },
        )

    with tab_browse:
        show_not_relevant = st.checkbox(f"Show '{NOT_RELEVANT}'", value=True)
        browse_labels = [l for l in ordered_labels if l != NOT_RELEVANT or show_not_relevant]

        if "selected_category" not in st.session_state:
            st.session_state.selected_category = browse_labels[0]
        if st.session_state.selected_category not in browse_labels:
            st.session_state.selected_category = browse_labels[0]

        col_cats, col_sentences = st.columns([2, 3], gap="large")

        with col_cats:
            st.markdown('<p class="panel-title">Categories</p>', unsafe_allow_html=True)

            cat_idx = browse_labels.index(st.session_state.selected_category)
            render_category_nav(browse_labels, cat_idx, "cat_top")

            for i, cat in enumerate(browse_labels):
                is_selected = cat == st.session_state.selected_category
                if render_category_row(cat, counts, len(results), is_selected, i):
                    if not is_selected:
                        st.session_state.selected_category = cat
                        st.session_state.page = 0
                        st.rerun()

            render_category_nav(
                browse_labels,
                browse_labels.index(st.session_state.selected_category),
                "cat_bottom",
            )

        with col_sentences:
            category = st.session_state.selected_category
            subset = results[results["new_category"] == category].copy()

            filt1, filt2 = st.columns([3, 1])
            with filt1:
                search = st.text_input("Search sentences", placeholder="Type a word...", key="sentence_search")
            with filt2:
                page_size = st.selectbox("Per page", [5, 10, 20, 50], index=1)

            if search.strip():
                mask = subset["origin_sentence"].str.contains(search.strip(), case=False, na=False)
                subset = subset[mask]

            st.html(
                f'<div class="category-header">'
                f'<span class="cat-name hebrew">{html_lib.escape(category)}</span>'
                f'<span class="count-he hebrew"> — {len(subset)} משפטים בקטגוריה</span>'
                f"</div>"
            )

            if subset.empty:
                st.info("No sentences found in this category.")
            else:
                total_pages = max(1, (len(subset) - 1) // page_size + 1)
                if "page" not in st.session_state:
                    st.session_state.page = 0
                if st.session_state.page >= total_pages:
                    st.session_state.page = 0

                pg_prev, pg_num, pg_next = st.columns([1, 2, 1])
                with pg_prev:
                    if st.button("Previous page", disabled=st.session_state.page == 0, key="pg_prev"):
                        st.session_state.page -= 1
                        st.rerun()
                with pg_num:
                    st.markdown(
                        f'<p class="nav-hint">Page {st.session_state.page + 1} of {total_pages}</p>',
                        unsafe_allow_html=True,
                    )
                with pg_next:
                    if st.button("Next page", disabled=st.session_state.page >= total_pages - 1, key="pg_next"):
                        st.session_state.page += 1
                        st.rerun()

                start = st.session_state.page * page_size
                for _, row in subset.iloc[start : start + page_size].iterrows():
                    render_sentence_card(row)

    with tab_prompt:
        render_prompt_panel(metadata)


if __name__ == "__main__":
    main()
