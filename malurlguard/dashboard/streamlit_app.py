"""Streamlit demo dashboard for MalUrlGuard."""

import sys
import time
from pathlib import Path
from typing import Optional

# Ensure package is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MalUrlGuard",
    page_icon="🛡️",
    layout="wide",
)

LABEL_COLORS = {
    "benign": "#28a745",
    "phishing": "#dc3545",
    "malware": "#8B0000",
    "defacement": "#fd7e14",
    "unknown": "#6c757d",
}

LABEL_ICONS = {
    "benign": "✅",
    "phishing": "🎣",
    "malware": "☠️",
    "defacement": "🖊️",
    "unknown": "❓",
}


# ── Cached predictor ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML models…")
def load_predictor():
    from malurlguard.inference.predictor import Predictor
    return Predictor()


@st.cache_resource(show_spinner="Loading SHAP explainer…")
def load_explainer():
    from malurlguard.inference.explainer import ShapExplainer
    return ShapExplainer()


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ MalUrlGuard")
    st.markdown("**ML-based Malicious URL Classifier**")
    st.markdown("---")
    st.markdown("**Target:** MBC Telkom University")
    st.markdown("**Phase:** 1 (RF + XGBoost)")
    st.markdown("---")
    show_explain = st.toggle("Show SHAP explanation", value=True)
    st.markdown("---")
    st.caption("Detects: phishing • malware • defacement • benign")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🛡️ MalUrlGuard — Malicious URL Classifier")
st.markdown(
    "Masukkan URL untuk diklasifikasikan menggunakan ensemble ML "
    "(Random Forest + XGBoost) dengan 30+ lexical features."
)

col1, col2 = st.columns([3, 1])
with col1:
    url_input = st.text_input(
        "URL to classify",
        placeholder="https://example.com/path?param=value",
        label_visibility="collapsed",
    )
with col2:
    classify_btn = st.button("🔍 Classify", use_container_width=True)

# Quick examples
st.markdown("**Quick examples:**")
example_cols = st.columns(4)
examples = [
    ("✅ Benign", "https://github.com/torvalds/linux"),
    ("🎣 Phishing", "http://paypal-verify-account.tk/login.php?id=123"),
    ("☠️ Malware", "http://185.220.101.5/update.exe?token=abc123"),
    ("🖊️ Defacement", "http://example.go.id/hacked-by-anonymous.html"),
]
for col, (label, example_url) in zip(example_cols, examples):
    if col.button(label, use_container_width=True):
        url_input = example_url
        classify_btn = True

st.markdown("---")

# ── Classification ────────────────────────────────────────────────────────────
if classify_btn and url_input:
    try:
        predictor = load_predictor()
        with st.spinner("Classifying…"):
            result = predictor.predict(url_input.strip())

        label = result["label"]
        confidence = result["confidence"]
        color = LABEL_COLORS.get(label, "#6c757d")
        icon = LABEL_ICONS.get(label, "❓")

        # Result banner
        st.markdown(
            f"""
            <div style="background:{color}22; border-left:5px solid {color};
                        padding:15px; border-radius:5px; margin:10px 0;">
              <h2 style="color:{color}; margin:0">{icon} {label.upper()}</h2>
              <p style="margin:5px 0 0">Confidence: <strong>{confidence:.1%}</strong>
                 &nbsp;|&nbsp; Latency: {result['latency_ms']} ms</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability bars
        st.markdown("#### Probability Breakdown")
        prob_data = {
            "Class": list(result["probabilities"].keys()),
            "Probability": list(result["probabilities"].values()),
        }
        df_prob = pd.DataFrame(prob_data).sort_values("Probability", ascending=False)
        st.bar_chart(df_prob.set_index("Class"))

        # SHAP explanation
        if show_explain:
            st.markdown("#### Top Feature Contributions (SHAP)")
            try:
                explainer = load_explainer()
                features = explainer.explain(url_input.strip())
                if features:
                    df_feat = pd.DataFrame(features)
                    df_feat["shap_value_display"] = df_feat["shap_value"].apply(
                        lambda x: f"{x:.4f}" if x is not None else "N/A"
                    )
                    for _, row in df_feat.iterrows():
                        direction_color = "#dc3545" if row["direction"] == "increases_risk" else "#28a745"
                        st.markdown(
                            f"<div style='padding:4px 0'>"
                            f"<b>{row['feature']}</b> = {row['value']} "
                            f"&nbsp;→&nbsp; SHAP: <span style='color:{direction_color}'>"
                            f"{row['shap_value_display']} ({row['direction']})</span></div>",
                            unsafe_allow_html=True,
                        )
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

        # Add to history
        st.session_state.history.insert(0, {
            "URL": url_input[:60] + ("…" if len(url_input) > 60 else ""),
            "Label": f"{icon} {label}",
            "Confidence": f"{confidence:.1%}",
            "ms": result["latency_ms"],
        })
        st.session_state.history = st.session_state.history[:10]

    except FileNotFoundError as exc:
        st.error(f"Models not found: {exc}\n\nRun: `malurlguard train --model all`")
    except Exception as exc:
        st.error(f"Error: {exc}")

elif classify_btn and not url_input:
    st.warning("Please enter a URL first.")

# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("#### Recent Predictions (last 10)")
    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True,
        hide_index=True,
    )
