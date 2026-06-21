"""Architecture page for the Direct-or-Web RAG Assistant."""

from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="Architecture | Direct-or-Web RAG Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui_markup_path = Path(__file__).resolve().parent.parent / "app_ui.html"
st.markdown(ui_markup_path.read_text(encoding="utf-8"), unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        '<nav class="app-navigation">'
        '<a href="/">💬 Assistant</a>'
        '<a class="active" href="/architecture">🏗️ Architecture</a>'
        "</nav>",
        unsafe_allow_html=True,
    )

st.title("Architecture")
st.write("Architecture details will be added here soon.")
