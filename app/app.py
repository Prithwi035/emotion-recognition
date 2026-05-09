"""
app.py
------
Streamlit frontend — UI only.
All detection logic lives in backend.py.
"""

import queue
import time

import cv2
import streamlit as st

from backend import (
    EMOTIONS,
    load_detector,
    open_camera,
    process_frame,
    pull_result,
    push_frame,
    start_worker,
)

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="centered"
)

# -----------------------------------
# CUSTOM STYLE
# -----------------------------------
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: #0d0d0d;
    }
    [data-testid="stSidebar"] {
        background: #111111;
        border-right: 1px solid #222;
    }
    [data-testid="stToolbar"] { display: none !important; }
    footer                    { display: none !important; }
    h1 {
        font-family: 'Courier New', monospace;
        font-size: 1.6rem !important;
        letter-spacing: 0.1em;
        color: #00ff88 !important;
    }
    .emotion-box {
        background: #111;
        border: 1px solid #00ff88;
        border-radius: 8px;
        padding: 12px 20px;
        font-family: 'Courier New', monospace;
        font-size: 1.3rem;
        color: #00ff88;
        margin-bottom: 10px;
    }
    .no-face-box {
        background: #111;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 12px 20px;
        font-family: 'Courier New', monospace;
        font-size: 1rem;
        color: #555;
        margin-bottom: 10px;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# TITLE
# -----------------------------------
st.title("😊 EMOTION RECOGNITION")

# -----------------------------------
# SIDEBAR CONTROLS
# -----------------------------------
st.sidebar.title("Controls")
run = st.sidebar.checkbox("Start Camera", value=True)
process_every_n = st.sidebar.slider(
    "Process Every N Frames",
    min_value=3,
    max_value=15,
    value=7,
    step=1,
    help="Higher = faster display, less frequent emotion updates"
)

# -----------------------------------
# CACHED RESOURCES (load once)
# -----------------------------------
@st.cache_resource
def get_detector():
    return load_detector()

@st.cache_resource
def get_camera():
    return open_camera()

@st.cache_resource
def get_queues():
    fq = queue.Queue(maxsize=1)
    rq = queue.Queue(maxsize=1)
    return fq, rq

@st.cache_resource
def get_worker(_detector, _frame_queue, _result_queue):
    return start_worker(_detector, _frame_queue, _result_queue)

detector                  = get_detector()
camera                    = get_camera()
frame_queue, result_queue = get_queues()
worker                    = get_worker(detector, frame_queue, result_queue)

# -----------------------------------
# UI PLACEHOLDERS
# -----------------------------------
frame_placeholder   = st.empty()
emotion_placeholder = st.empty()

st.sidebar.subheader("Emotion Scores")
sidebar_slots = {e: st.sidebar.empty() for e in EMOTIONS}
fps_slot = st.sidebar.empty()

# -----------------------------------
# STATE
# -----------------------------------
frame_count = 0
last_result = {
    "emotion":    "no_face",
    "confidence": 0.0,
    "emotions":   {e: 0.0 for e in EMOTIONS},
    "box":        None,
}

# -----------------------------------
# MAIN LOOP
# -----------------------------------
while run:
    t0 = time.perf_counter()

    ret, frame = camera.read()
    if not ret:
        st.error("Cannot access webcam. Check camera connection.")
        break

    frame_count += 1

    if frame_count % process_every_n == 0:
        push_frame(frame_queue, frame)

    last_result = pull_result(result_queue, last_result)
    no_face     = last_result.get("emotion") == "no_face"

    # Annotate and display frame
    display_rgb = process_frame(frame, last_result)
    frame_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

    # Emotion label
    if no_face:
        emotion_placeholder.markdown(
            "<div class='no-face-box'>👤 &nbsp; No face detected</div>",
            unsafe_allow_html=True
        )
    else:
        emotion_placeholder.markdown(
            f"<div class='emotion-box'>"
            f"🎭 &nbsp; <b>{last_result['emotion'].upper()}</b>"
            f" &nbsp;|&nbsp; confidence: {last_result['confidence']:.2f}"
            f"</div>",
            unsafe_allow_html=True
        )

    # Sidebar scores — only show when face detected
    for emotion, slot in sidebar_slots.items():
        if no_face:
            slot.markdown(f"`{emotion:<9}` `—`")
        else:
            score = last_result["emotions"].get(emotion, 0.0)
            bar   = "█" * int(score * 20)
            slot.markdown(f"`{emotion:<9}` {bar} `{score:.2f}`")

    # FPS
    fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
    fps_slot.markdown(f"**FPS:** `{fps:.1f}`")

# -----------------------------------
# CLEANUP
# -----------------------------------
frame_queue.put(None)
camera.release()
st.info("Camera stopped.")
