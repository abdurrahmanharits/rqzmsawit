"""Entrypoint: `streamlit run Home.py`.

Each page under `page/` calls its own `st.set_page_config(...)`, so this
entrypoint intentionally does not call it — st.navigation runs exactly one
page script per request.
"""
from __future__ import annotations

import streamlit as st

pages = [
    st.Page("page/okta_sawit.py", title="Kebun Sawit", icon="🌴", default=True),
]

pg = st.navigation(pages)
pg.run()
