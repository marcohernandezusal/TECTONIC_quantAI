# Home.py – landing page that pulls content from the quantAI API
from __future__ import annotations

import os
from pathlib import Path

import httpx
import streamlit as st

###############################################################################
# Configuration                                                               #
###############################################################################
API_URL = os.getenv("API_URL", "http://localhost:8000")  # e.g. http://myhost:8000
BACKEND_README_ENDPOINT = "/readme"                      # customise if needed
LOCAL_README = "README_app.md"

###############################################################################
# UI header                                                                   #
###############################################################################
st.set_page_config(page_title="TECTONIC quantAI – Home", layout="centered")
st.title("TECTONIC quantAI  🛠️")
st.subheader("Corrosion Prediction for Underwater Cultural Heritage")

###############################################################################
# Contact backend: health-check & fetch README                                #
###############################################################################
readme_md: str | None = None
backend_ok = False

with st.spinner("Contacting backend …"):
    try:
        with httpx.Client(base_url=API_URL, timeout=10) as client:
            # 1️⃣ simple health check (optional but nice to show status)
            health = client.get("/health")
            health.raise_for_status()
            backend_ok = True
            st.success(f"Backend online ({API_URL})")

            # 2️⃣ fetch README from the API (plain text/markdown response)
            r = client.get(BACKEND_README_ENDPOINT)
            if r.is_success:
                readme_md = r.text
    except Exception as exc:
        st.warning(f"Could not reach backend at {API_URL} – {exc}")

###############################################################################
# Fallback to local file if remote README not available                       #
###############################################################################
if readme_md is None:
    readme_path = Path(__file__).with_name(LOCAL_README)
    if readme_path.exists():
        readme_md = readme_path.read_text(encoding="utf-8")
        st.info("Loaded README from local file.")
    else:
        st.error(f"Neither backend nor local {LOCAL_README} found.")
        st.stop()

###############################################################################
# Patch /static/ links so they hit the backend port                           #
###############################################################################
readme_md = readme_md.replace("](/static/", f"]({API_URL}/static/")

###############################################################################
# Render                                                                      #
###############################################################################
st.markdown(readme_md, unsafe_allow_html=True)

st.info("Use the sidebar to open **Quant models** and run your predictions.")
