import os
import json
from typing import Any, Dict, List

import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# Basic page config & light styling
# -----------------------------
st.set_page_config(
    page_title="IA Assistant (Local)",
    page_icon="🤖",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.small { font-size: 0.85rem; color: #6b7280; }
.card { background: var(--background-color, #0e1117); border: 1px solid rgba(148,163,184,.2); padding: 1rem; border-radius: 16px; }
.kpi { font-size: 0.9rem; color: #6b7280; margin-top: .25rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Sidebar controls
# -----------------------------
DEFAULT_API = os.getenv("API_BASE", "http://localhost:8080")
with st.sidebar:
    st.header("⚙️ API")
    api_base = st.text_input("API base URL", value=DEFAULT_API, help="Adresse FastAPI (ex: http://localhost:8080)")
    st.divider()

    st.header("🔎 Recherche")
    days = st.slider("Fenêtre (jours)", min_value=0, max_value=90, value=7, step=1)
    top_k = st.slider("Top K (passages)", min_value=1, max_value=50, value=20, step=1,
                      help="Nombre de passages vectoriels retournés")
    want_chart = st.checkbox("Inclure un graphique de synthèse", value=True)

    st.divider()
    st.caption("Astuce: modifiez l'URL API si l'appli tourne sur une autre machine/port.")

# -----------------------------
# Header
# -----------------------------
st.title("🤖 SOC Assistant — Malcolm RAG (Local)")
st.write(
    "Posez une question en français sur vos journaux Malcolm (Zeek/Suricata/Arkime).\n"
    "Le modèle répond au format **Constat → Analyse → Impact → Recommandations**."
)

# -----------------------------
# Input area
# -----------------------------
example_q = "Quelles ...?"
question = st.text_area("Votre question", value=example_q, height=100)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    ask_clicked = st.button("Interroger l'IA", type="primary")
with col_info:
    st.markdown("<div class='small'>Backend : <code>/ask</code> sur votre FastAPI</div>", unsafe_allow_html=True)

# -----------------------------
# Call API
# -----------------------------
def call_ask(api_base: str, question: str, days: int, top_k: int, want_chart: bool) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/ask"
    payload = {
        "question": question,
        "days": int(days),
        "top_k": int(top_k),
        "want_chart": bool(want_chart),
    }
    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    return resp.json()

# -----------------------------
# Results zone
# -----------------------------
if ask_clicked:
    if not question or len(question.strip()) < 3:
        st.warning("Veuillez saisir une question plus précise.")
    else:
        with st.spinner("Analyse en cours…"):
            try:
                data = call_ask(api_base, question.strip(), days, top_k, want_chart)
            except Exception as e:
                st.error(f"Erreur d'appel API : {e}")
                st.stop()

        # Answer card
        st.subheader("🧠 Réponse de l'IA")
        st.markdown(f"<div class='card'>{data.get('answer','(réponse vide)')}</div>", unsafe_allow_html=True)

        # Sources table
        used_sources: List[Dict[str, Any]] = data.get("used_sources", []) or []
        if used_sources:
            st.subheader("📚 Passages utilisés (vecteur)")
            # Flatten a bit for display
            rows = []
            for it in used_sources:
                rows.append({
                    "score": round(float(it.get("vector_hit_score", 0.0)), 4),
                    "index": it.get("_index", ""),
                    "_id": it.get("_id", ""),
                    "aperçu": (it.get("text", "")[:180] + "…") if it.get("text") else "",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune source vectorielle (l'index FAISS est peut‑être vide).")

        # Chart
        chart = data.get("chart")
        if chart and isinstance(chart, dict):
            st.subheader("📊 Synthèse graphique")
            if chart.get("type") == "bar":
                x = chart.get("x", [])
                y = chart.get("y", [])
                fig = px.bar(x=x, y=y, labels={"x": chart.get("x_title", "X"), "y": chart.get("y_title", "Y")},
                             title=chart.get("title", ""))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.json(chart)

        # Raw JSON (optional debug expander)
        with st.expander("🔧 Détails bruts (debug)"):
            st.json(data)

# Footer
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("Entièrement local : OpenSearch + FAISS + SentenceTransformers + Ollama (Mistral)")
