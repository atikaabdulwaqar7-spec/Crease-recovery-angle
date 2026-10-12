# Crease Recovery Angle Analyzer

A Streamlit web app that measures the **crease recovery angle** of a fabric specimen from a photograph.

## How it works
1. **Segmentation**: the fabric is separated from the background (Otsu / adaptive / manual threshold).
2. **Arm detection**: Canny edges + Hough transform find straight segments; their orientations are clustered into two groups (the two arms of the fold).
3. **Angle**: a line is fitted to each arm, the lines are intersected to find the crease vertex, and the angle between the arms is measured from that vertex.
4. **Manual mode**: click the crease and the two arm tips if automatic detection struggles.
5. Save multiple readings, label them Warp/Weft, and download a CSV with mean, standard deviation and total (warp + weft) recovery angle.

## Project structure
```
app.py                # Streamlit interface
image_processing.py   # loading, resizing, segmentation
angle_detection.py    # auto + manual angle calculation
utils.py              # drawing and result summary helpers
credits.py            # team names / roll numbers shown in the app (edit this)
requirements.txt
```

## Team credits
Open `credits.py` to edit the team names, roll numbers and university. They appear in a banner at the top, in the sidebar and in the footer of the app.

## Run locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

## Tips for good photos
- Plain, contrasting background (white specimen on dark, or the reverse).
- Camera directly above / in front of the fold, not tilted.
- Even lighting, no strong shadows.
- Take the photo after the standard recovery time (e.g. 5 min as per AATCC 66 / ISO 2313).

## Deploy
Push to GitHub, then deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud) by pointing it at `app.py`.
