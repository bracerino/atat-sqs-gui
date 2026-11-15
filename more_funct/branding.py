"""Supporter logos at the bottom of the sidebar (same block as in XRDlicious)."""

import base64
from pathlib import Path

import streamlit as st

_IMAGES = Path(__file__).resolve().parent.parent / "images_atat_gui"


def _data_uri(path):
    mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


# (name, website, logo file, logo height in px)
SUPPORTERS = [
    ("FZU – Institute of Physics of the Czech Academy of Sciences",
     "https://www.fzu.cz/en", "fzu_logo.svg", 34),
    ("IMPLANT group", "https://implant.fs.cvut.cz/", "implant_logo.png", 44),
]


def render_supporters(container):
    logos = []
    for name, url, fname, height in SUPPORTERS:
        path = _IMAGES / fname
        if not path.exists():
            continue
        logos.append(
            f'<a href="{url}" target="_blank" title="{name}">'
            f'<img src="{_data_uri(path)}" alt="{name}" '
            f'style="height:{height}px;width:auto;"></a>')
    # Keyed so Streamlit tracks the block by key, not by position: otherwise a
    # rerun that adds sidebar content above it briefly shows the old and the
    # new copy at once.
    box = container.container(key="sidebar_supporters")
    box.markdown("---")
    box.markdown(
        '<div style="font-size:0.85em;opacity:0.8;margin-bottom:6px;">'
        'Supported by <b>FZU</b> and the <b>IMPLANT group</b></div>'
        '<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">'
        + "".join(logos) + '</div>',
        unsafe_allow_html=True,
    )
