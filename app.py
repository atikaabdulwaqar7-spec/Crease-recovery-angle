"""
app.py  -  Crease Recovery Angle Analyzer
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

from image_processing import load_image, resize_max, segment_fabric
from cra_analysis import detect_angle_auto, manual_result
from utils import annotate, draw_points, bgr_to_rgb, summarize
from credits import render_banner, render_sidebar_credits, render_footer

st.set_page_config(page_title="Crease Recovery Angle Analyzer", page_icon="🧵", layout="wide")

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = []      # list of dict rows
if "points" not in st.session_state:
    st.session_state.points = []       # manual clicks
if "last_click" not in st.session_state:
    st.session_state.last_click = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("Settings")
sample_name = st.sidebar.text_input("Sample name", "Sample 1")
direction = st.sidebar.selectbox("Direction", ["Warp", "Weft"])

st.sidebar.subheader("Segmentation (auto mode)")
seg_method = st.sidebar.selectbox("Threshold method", ["otsu", "adaptive", "manual"])
threshold = st.sidebar.slider("Manual threshold", 0, 255, 127, disabled=seg_method != "manual")
polarity = st.sidebar.selectbox("Fabric vs background", ["auto", "fabric_bright", "fabric_dark"])
blur = st.sidebar.slider("Blur kernel", 1, 15, 5, step=2)
morph = st.sidebar.slider("Noise cleanup", 0, 15, 5)

st.sidebar.subheader("Line detection (auto mode)")
min_line_frac = st.sidebar.slider("Min line length (% of image)", 5, 40, 12) / 100
hough_thr = st.sidebar.slider("Hough threshold", 10, 120, 40)

render_sidebar_credits()   # team names + roll numbers in the sidebar

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🧵 Crease Recovery Angle Analyzer")
render_banner()            # team card at the top of the page
st.caption(
    "Upload a photo of the creased fabric specimen taken straight-on (camera perpendicular to the "
    "fold line) after the recovery time. The app measures the angle between the two arms of the fold."
)

uploaded = st.file_uploader("Upload specimen image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

if uploaded is None:
    st.info("Upload an image to begin.")
    render_footer()
    st.stop()

# Reset manual points when a new file is loaded
if st.session_state.current_file != uploaded.name:
    st.session_state.current_file = uploaded.name
    st.session_state.points = []
    st.session_state.last_click = None

img_full = load_image(uploaded)
img, _ = resize_max(img_full, 1000)

tab_auto, tab_manual, tab_results = st.tabs(["Automatic", "Manual (3 clicks)", "Results"])


def save_result(res):
    st.session_state.results.append(
        {
            "Sample": sample_name,
            "Direction": direction,
            "Angle (deg)": round(res.angle, 2),
            "Method": res.method,
            "Image": uploaded.name,
        }
    )
    st.success(f"Saved {res.angle:.2f}° for {sample_name} ({direction}).")


# ----------------------------------------------------------------------------
# Automatic tab
# ----------------------------------------------------------------------------
with tab_auto:
    mask = segment_fabric(
        img, blur=blur, method=seg_method, threshold=threshold, polarity=polarity, morph=morph
    )
    res = detect_angle_auto(mask, min_line_frac=min_line_frac, hough_threshold=hough_thr)

    c1, c2 = st.columns(2)
    c1.image(bgr_to_rgb(img), caption="Original", use_container_width=True)
    c2.image(mask, caption="Segmented mask", use_container_width=True)

    if res is None:
        st.warning(
            "Could not find two distinct arms. Try changing the fabric/background setting, lowering "
            "the Hough threshold or minimum line length, or use the Manual tab."
        )
    else:
        st.image(bgr_to_rgb(annotate(img, res)), caption="Detected arms", use_container_width=True)
        st.metric("Crease recovery angle", f"{res.angle:.2f}°")
        if st.button("Save this result", key="save_auto"):
            save_result(res)

# ----------------------------------------------------------------------------
# Manual tab
# ----------------------------------------------------------------------------
with tab_manual:
    st.write(
        "Click **1) the crease (vertex)**, **2) the tip of arm one**, **3) the tip of arm two**."
    )
    disp_w = 800
    ih, iw = img.shape[:2]
    disp = Image.fromarray(bgr_to_rgb(draw_points(img, st.session_state.points)))
    disp = disp.resize((disp_w, int(ih * disp_w / iw)))

    click = streamlit_image_coordinates(disp, key="manual_click")
    if click and click != st.session_state.last_click and len(st.session_state.points) < 3:
        st.session_state.last_click = click
        x = click["x"] * iw / click["width"]
        y = click["y"] * ih / click["height"]
        st.session_state.points.append((x, y))
        st.rerun()

    st.write(f"Points selected: {len(st.session_state.points)} / 3")
    if st.button("Reset points"):
        st.session_state.points = []
        st.session_state.last_click = None
        st.rerun()

    if len(st.session_state.points) == 3:
        v, p1, p2 = st.session_state.points
        mres = manual_result(v, p1, p2)
        st.metric("Crease recovery angle", f"{mres.angle:.2f}°")
        if st.button("Save this result", key="save_manual"):
            save_result(mres)

# ----------------------------------------------------------------------------
# Results tab
# ----------------------------------------------------------------------------
with tab_results:
    if not st.session_state.results:
        st.info("No saved measurements yet.")
    else:
        df = pd.DataFrame(st.session_state.results)
        st.dataframe(df, use_container_width=True)
        st.subheader("Summary")
        st.dataframe(summarize(df), use_container_width=True)
        st.download_button(
            "Download results (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="crease_recovery_results.csv",
            mime="text/csv",
        )
        if st.button("Clear all results"):
            st.session_state.results = []
            st.rerun()

# Footer credits (shown after the last tab)
render_footer()
