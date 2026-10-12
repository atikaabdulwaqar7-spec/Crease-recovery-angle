"""
credits.py
----------
Project credits shown to everyone who opens the app.

>>> EDIT ONLY THE SETTINGS BELOW <<<  (names, roll numbers, university)
"""

import streamlit as st

# ----------------------------------------------------------------------------
# EDIT THESE
# ----------------------------------------------------------------------------
PROJECT_TITLE = "Crease Recovery Angle Analyzer"
UNIVERSITY = "Mehran University of Engineering & Technology, Jamshoro"
DEPARTMENT = "Department of Textile Engineering"
YEAR = "2026"

TEAM = [
    {"name": "Ateeqa Waqar", "roll_no": "24TE013"},
    {"name": "Eeman Khan", "roll_no": "24TE037"},
    {"name": "Muhammad Safiullah", "roll_no": "24TE033"},
]
# ----------------------------------------------------------------------------


def render_banner():
    """Team card shown at the top of the page (collapsible, open by default)."""
    with st.expander("About this project / Developed by", expanded=True):
        st.markdown(f"**{PROJECT_TITLE}**  \n{DEPARTMENT}, {UNIVERSITY}")
        cols = st.columns(len(TEAM))
        for col, member in zip(cols, TEAM):
            col.markdown(f"**{member['name']}**  \nRoll No: `{member['roll_no']}`")


def render_sidebar_credits():
    """Compact credits at the bottom of the sidebar."""
    st.sidebar.divider()
    st.sidebar.markdown("**Developed by**")
    for member in TEAM:
        st.sidebar.markdown(f"{member['name']}  \n`{member['roll_no']}`")
    st.sidebar.caption(f"{DEPARTMENT}  \n{UNIVERSITY}")


def render_footer():
    """One-line footer at the very bottom of the page."""
    names = " • ".join(f"{m['name']} ({m['roll_no']})" for m in TEAM)
    st.divider()
    st.caption(f"© {YEAR} {names} — {UNIVERSITY}")
