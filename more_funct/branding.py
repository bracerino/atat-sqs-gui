"""Supporter logos and journal badge at the bottom of the sidebar (same block as in XRDlicious)."""

import base64
from pathlib import Path

import streamlit as st

_IMAGES = Path(__file__).resolve().parent.parent / "images_atat_gui"


def _data_uri(path):
    mime = {".svg": "image/svg+xml", ".webp": "image/webp",
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(path.suffix.lower(), "image/png")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


# (name, website, logo file, logo height in px)
SUPPORTERS = [
    ("FZU – Institute of Physics of the Czech Academy of Sciences",
     "https://www.fzu.cz/en", "fzu_logo.svg", 34),
    ("IMPLANT group", "https://implant.fs.cvut.cz/", "implant_logo.png", 44),
]


# The journal SimplySQS is published in; the badge links to the article. A journal
# logo saved as images_atat_gui/jocs_logo.(svg|png|webp) replaces the whole badge;
# otherwise the journal name is drawn as text, next to the cover thumbnail
# (images_atat_gui/jocs_cover.jpg) when that file is present.
ARTICLE_URL = "https://doi.org/10.1016/j.jocs.2026.102846"
ARTICLE_TITLE = ("M. Lebeda et al. (2026). SimplySQS: An automated and reproducible workflow "
                 "for special quasirandom structure generation with ATAT. "
                 "Journal of Computational Science")
JOURNAL_LOGO_FILES = ("jocs_logo.svg", "jocs_logo.png", "jocs_logo.webp")
JOURNAL_COVER_FILES = ("jocs_cover.jpg", "jocs_cover.png", "jocs_cover.webp")


def _first_existing(names):
    return next((_IMAGES / f for f in names if (_IMAGES / f).exists()), None)


def _journal_html():
    logo = _first_existing(JOURNAL_LOGO_FILES)
    if logo is not None:
        badge = (f'<img src="{_data_uri(logo)}" alt="Journal of Computational Science" '
                 'style="max-height:44px;max-width:100%;width:auto;">')
    else:
        cover = _first_existing(JOURNAL_COVER_FILES)
        cover_html = (f'<img src="{_data_uri(cover)}" alt="Journal of Computational Science cover" '
                      'style="height:56px;width:auto;flex:none;border-radius:2px;'
                      'box-shadow:0 1px 3px rgba(0,0,0,0.3);">' if cover else "")
        # without the cover, a coloured bar on the left takes its place
        bar = "" if cover else "border-left:4px solid #e9711c;padding-left:8px;"
        badge = (
            '<span style="display:inline-flex;align-items:center;gap:10px;">'
            + cover_html +
            f'<span style="display:inline-block;{bar}padding-top:2px;padding-bottom:2px;'
            'line-height:1.05;font-family:Georgia,\'Times New Roman\',serif;color:#1f4e8c;">'
            '<span style="font-size:0.78em;font-style:italic;">Journal of</span><br>'
            '<span style="font-size:0.95em;font-weight:700;white-space:nowrap;">'
            'Computational Science</span></span></span>')
    return (
        '<div style="font-size:0.85em;opacity:0.8;margin:14px 0 6px 0;">Published in</div>'
        f'<a href="{ARTICLE_URL}" target="_blank" title="{ARTICLE_TITLE}" '
        'style="text-decoration:none;">' + badge + '</a>')


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
        + "".join(logos) + '</div>' + _journal_html(),
        unsafe_allow_html=True,
    )
