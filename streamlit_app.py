"""Entry point for `streamlit run streamlit_app.py` and Streamlit Community Cloud."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402

# Streamlit Cloud: copy top-level secrets into env vars so pydantic Settings picks them up.
try:
    for key, value in st.secrets.items():
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(key.upper(), str(value))
except Exception:
    pass  # no secrets file locally -> use .env

from rag_agent.ui.chat import main  # noqa: E402

main()
