"""Visual styling for the developer-tool look."""
import streamlit as st

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
h1, h2, h3 {letter-spacing: -0.01em;}
[data-testid="stMetric"] {background:#161B22; border:1px solid #30363D; border-radius:10px; padding:12px 16px;}
.rp-badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; margin-right:6px;}
.rp-green {background:#12361f; color:#56d364; border:1px solid #238636;}
.rp-amber {background:#3a2d0b; color:#e3b341; border:1px solid #9e6a03;}
.rp-red   {background:#3d1418; color:#ff7b72; border:1px solid #da3633;}
.rp-blue  {background:#0d2a52; color:#79c0ff; border:1px solid #1f6feb;}
.rp-note {border-left:3px solid #4F8CFF; padding:6px 12px; background:#161B22; border-radius:4px; font-size:0.9rem;}
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(text: str, color: str = "blue") -> str:
    """HTML badge (green/amber/red/blue). Text is escaped by callers passing constants only."""
    return f'<span class="rp-badge rp-{color}">{text}</span>'


def note(text: str) -> None:
    st.markdown(f'<div class="rp-note">{text}</div>', unsafe_allow_html=True)
