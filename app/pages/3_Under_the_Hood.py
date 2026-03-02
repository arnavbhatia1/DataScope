"""
MarketPulse — Under the Hood

Diagnostics page exposing ML internals: model metadata, feature importance,
labeling function performance, and data quality metrics.
"""

import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on path.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.pipeline_runner import load_model, run_pipeline
from app.components.styles import COLORS, SENTIMENT_COLORS, apply_theme
from src.labeling.quality import LabelQualityAnalyzer
from src.labeling.functions import LABELING_FUNCTIONS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Under the Hood | MarketPulse",
    page_icon="M",
    layout="wide",
)
apply_theme()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Under the Hood")
st.markdown("ML diagnostics: model internals, labeling function performance, and data quality metrics.")

# ---------------------------------------------------------------------------
# Load model and pipeline data (both cached)
# ---------------------------------------------------------------------------
model = load_model()

if model is None:
    st.warning(
        "No trained model found. Run the pipeline first.\n\n"
        "From the project root: `make train` or `python scripts/train.py`"
    )
    st.stop()

with st.spinner("Loading pipeline data..."):
    data = run_pipeline(
        start_date_str=st.session_state.get("start_date"),
        end_date_str=st.session_state.get("end_date"),
    )

df = data.get("df", pd.DataFrame())

# ---------------------------------------------------------------------------
# Section 1 — Model Metadata
# ---------------------------------------------------------------------------
st.header("Model Metrics")

metadata_path = os.path.join(_project_root, "data", "models", "model_metadata.json")
metadata = {}
try:
    with open(metadata_path, "r") as fh:
        metadata = json.load(fh)
except FileNotFoundError:
    st.info("model_metadata.json not found — showing metadata from loaded model object.")
    metadata = getattr(model, "metadata", {})
except Exception as exc:
    st.warning(f"Could not read model_metadata.json: {exc}")
    metadata = getattr(model, "metadata", {})

# Surface key metrics
metrics_block = metadata.get("metrics", {})
val_metrics = metrics_block.get("validation_metrics", {})
cv_block = metrics_block.get("cross_val_scores") or {}

val_f1 = val_metrics.get("weighted_f1") or val_metrics.get("f1")
val_acc = val_metrics.get("accuracy")
num_features = metadata.get("num_features") or metrics_block.get("num_features")
training_date = metadata.get("training_date", "Unknown")
classes = metadata.get("classes") or metrics_block.get("classes", [])
dataset_size = metadata.get("dataset_size")
cv_mean = cv_block.get("mean") if cv_block else None
cv_std = cv_block.get("std") if cv_block else None

meta_cols = st.columns(4)
with meta_cols[0]:
    st.metric(
        "Validation F1",
        f"{val_f1:.3f}" if val_f1 is not None else "N/A",
    )
with meta_cols[1]:
    st.metric(
        "Validation Accuracy",
        f"{val_acc:.1%}" if val_acc is not None else "N/A",
    )
with meta_cols[2]:
    st.metric(
        "Feature Vocabulary",
        f"{num_features:,}" if num_features is not None else "N/A",
    )
with meta_cols[3]:
    st.metric(
        "Training Set Size",
        f"{dataset_size:,}" if dataset_size is not None else "N/A",
    )

detail_cols = st.columns(2)
with detail_cols[0]:
    if cv_mean is not None:
        st.metric(
            "Cross-Val F1 (mean ± std)",
            f"{cv_mean:.3f} ± {cv_std:.3f}" if cv_std is not None else f"{cv_mean:.3f}",
        )
with detail_cols[1]:
    if classes:
        st.markdown(f"**Classes trained on:** {', '.join(c.upper() for c in classes)}")
    if training_date and training_date != "Unknown":
        ts = training_date[:19].replace("T", " ")
        st.markdown(f"**Training date:** `{ts}`")

# Per-class validation metrics table
if "classification_report" in val_metrics:
    per_class_report = val_metrics["classification_report"]
    rows = []
    for cls in ["bullish", "bearish", "neutral", "meme"]:
        if cls in per_class_report:
            entry = per_class_report[cls]
            rows.append({
                "Class": cls.upper(),
                "Precision": f"{entry.get('precision', 0):.3f}",
                "Recall": f"{entry.get('recall', 0):.3f}",
                "F1": f"{entry.get('f1-score', 0):.3f}",
                "Support": int(entry.get("support", 0)),
            })
    if rows:
        with st.expander("Per-class validation metrics", expanded=False):
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 2 — Feature Importance
# ---------------------------------------------------------------------------
st.divider()
st.header("Feature Importance")
st.caption("Top TF-IDF features driving each class prediction, ranked by logistic regression coefficient.")

try:
    importance = model.get_feature_importance(top_n=10)
except Exception as exc:
    importance = {}
    st.warning(f"Could not retrieve feature importance: {exc}")

if importance:
    class_order = [c for c in ["bullish", "bearish", "neutral", "meme"] if c in importance]
    fi_cols = st.columns(len(class_order))

    for col_idx, cls in enumerate(class_order):
        features_coefs = importance[cls]
        features = [f for f, _ in features_coefs]
        coefs = [c for _, c in features_coefs]
        bar_color = SENTIMENT_COLORS.get(cls, COLORS["secondary"])

        fig = go.Figure(data=[go.Bar(
            x=coefs,
            y=features,
            orientation="h",
            marker_color=bar_color,
            text=[f"{c:.3f}" for c in coefs],
            textposition="auto",
        )])
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text=cls.upper(), font=dict(color=bar_color, size=14)),
            height=320,
            margin=dict(t=40, b=20, l=100, r=20),
            xaxis_title="Coefficient",
            yaxis=dict(autorange="reversed"),
            showlegend=False,
        )

        with fi_cols[col_idx]:
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3 — Labeling Function Performance
# ---------------------------------------------------------------------------
st.divider()
st.header("Labeling Function Performance")
st.caption(
    "Each function's coverage (% of posts it voted on), conflict rate, and gold accuracy "
    "(when gold labels are available)."
)

if df.empty:
    st.info("No pipeline data available. Run the pipeline to see labeling function metrics.")
else:
    with st.spinner("Computing per-function report..."):
        try:
            analyzer = LabelQualityAnalyzer(LABELING_FUNCTIONS)
            lf_report = analyzer.per_function_report(df)
        except Exception as exc:
            lf_report = pd.DataFrame()
            st.warning(f"Could not compute labeling function report: {exc}")

    if not lf_report.empty:
        # Format for display
        display_lf = lf_report[["function", "coverage", "num_voted", "conflict_rate"]].copy()
        display_lf["coverage"] = display_lf["coverage"].apply(lambda v: f"{v:.1%}")
        display_lf["conflict_rate"] = display_lf["conflict_rate"].apply(lambda v: f"{v:.1%}")
        display_lf.columns = ["Function", "Coverage", "Votes Cast", "Conflict Rate"]

        # Include gold accuracy column if available
        if "accuracy_on_gold" in lf_report.columns and lf_report["accuracy_on_gold"].notna().any():
            display_lf["Gold Accuracy"] = lf_report["accuracy_on_gold"].apply(
                lambda v: f"{v:.1%}" if pd.notna(v) else "—"
            )

        st.dataframe(display_lf, use_container_width=True, hide_index=True)

        # Coverage bar chart
        with st.expander("Coverage chart by function", expanded=False):
            raw_coverage = lf_report["coverage"].tolist()
            func_names = lf_report["function"].tolist()

            fig_cov = go.Figure(data=[go.Bar(
                x=raw_coverage,
                y=func_names,
                orientation="h",
                marker_color=COLORS["primary"],
                text=[f"{v:.1%}" for v in raw_coverage],
                textposition="auto",
            )])
            fig_cov.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title="Labeling Function Coverage",
                height=max(300, len(func_names) * 28),
                margin=dict(t=40, b=20, l=200, r=40),
                xaxis=dict(tickformat=".0%", range=[0, 1]),
                xaxis_title="Coverage",
            )
            st.plotly_chart(fig_cov, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4 — Data Quality
# ---------------------------------------------------------------------------
st.divider()
st.header("Data Quality")
st.caption("Aggregate statistics on programmatic label coverage, confidence, and conflicts.")

if df.empty:
    st.info("No pipeline data available. Run the pipeline to see data quality metrics.")
else:
    with st.spinner("Computing quality report..."):
        try:
            if "analyzer" not in dir():
                analyzer = LabelQualityAnalyzer(LABELING_FUNCTIONS)
            quality = analyzer.aggregate_quality_report(df)
        except Exception as exc:
            quality = {}
            st.warning(f"Could not compute quality report: {exc}")

    if quality:
        dq_cols = st.columns(5)
        dq_cols[0].metric("Total Posts", f"{quality.get('total_posts', 0):,}")
        dq_cols[1].metric("Labeled Posts", f"{quality.get('labeled_posts', 0):,}")
        dq_cols[2].metric(
            "Coverage",
            f"{quality.get('total_coverage', 0):.1%}",
        )
        dq_cols[3].metric(
            "Conflict Rate",
            f"{quality.get('conflict_rate', 0):.1%}",
        )
        dq_cols[4].metric(
            "Avg Votes / Post",
            f"{quality.get('avg_votes_per_post', 0):.1f}",
        )

        # Confidence distribution histogram
        conf_stats = quality.get("confidence_distribution", {})
        if conf_stats and "label_confidence" in df.columns:
            conf_values = df["label_confidence"].dropna().tolist()
            if conf_values:
                fig_conf = go.Figure(data=[go.Histogram(
                    x=conf_values,
                    nbinsx=20,
                    marker_color=COLORS["primary"],
                    opacity=0.85,
                    name="Confidence",
                )])
                fig_conf.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    title="Label Confidence Distribution",
                    height=280,
                    margin=dict(t=40, b=40, l=50, r=20),
                    xaxis_title="Confidence Score",
                    yaxis_title="Post Count",
                    bargap=0.05,
                )
                st.plotly_chart(fig_conf, use_container_width=True)

        # Label distribution breakdown
        label_dist = quality.get("label_distribution", {})
        if label_dist:
            with st.expander("Final label distribution", expanded=False):
                ld_rows = [
                    {
                        "Sentiment": k.upper(),
                        "Count": v,
                        "Share": f"{v / max(sum(label_dist.values()), 1):.1%}",
                    }
                    for k, v in sorted(label_dist.items(), key=lambda x: -x[1])
                ]
                st.dataframe(pd.DataFrame(ld_rows), hide_index=True, use_container_width=True)

        # Uncertain posts table
        uncertain_count = quality.get("uncertain_count", 0)
        if uncertain_count > 0 and "programmatic_label" in df.columns:
            with st.expander(f"Uncertain posts ({uncertain_count} without a confident label)", expanded=False):
                uncertain_df = df[df["programmatic_label"].isna()][["text", "label_confidence"]].copy()
                if "label_confidence" in uncertain_df.columns:
                    uncertain_df = uncertain_df.sort_values("label_confidence")
                uncertain_df = uncertain_df.head(50).reset_index(drop=True)
                st.dataframe(uncertain_df, use_container_width=True, hide_index=True)
                st.caption("These posts are prime candidates for human review or additional labeling functions.")
