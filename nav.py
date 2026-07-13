"""Sidebar navigation shared by all pages under `page/`."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

PAGES = [
    {"path": "page/okta_sawit.py", "label": "Dashboard RQZM Sawit", "icon": "🌴"},
]


def render_sidebar(current_file: str | Path) -> None:
    current_name = Path(current_file).name
    with st.sidebar:
        st.markdown("### RQZM-OKTA Dashboard")
        for p in PAGES:
            label = f"{p['icon']} {p['label']}"
            if Path(p["path"]).name == current_name:
                st.markdown(f"**➡ {label}**")
            else:
                try:
                    st.page_link(p["path"], label=label, icon=p["icon"])
                except Exception:
                    st.caption(label)
        st.markdown("---")
