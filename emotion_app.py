import streamlit as st
from fer import FER
import cv2
import time
import threading
import queue
import numpy as np

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Real-Time Emotion Recognition")

# -----------------------------------
# SIDEBAR CONTROLS
# -----------------------------------
st.sidebar.title("Controls")
run = st.sidebar.checkbox("Start Camera", value=True)

sad_boost = 1.20 
process_every_n = st.sidebar.slider(
    "Process Every N Frames",
    min_value=3,
    max_value=15,
    value=7,
    step=1,
    help="Higher = faster display, less frequent emotion updates"
)

# -----------------------------------
# LOAD DETECTOR (cached)
# -----------------------------------
@st.cache_resource
def load_detector():
    return FER(mtcnn=False)  # mtcnn=False is faster than default

detector = load_detector()

# -----------------------------------
# SHARED STATE (thread-safe)
# -----------------------------------
result_queue = queue.Queue(maxsize=1)  # Only keep latest result

# -----------------------------------
# EMOTION PROCESSING THREAD
# Runs inference off the main display loop
# -----------------------------------
def emotion_worker(frame_queue: queue.Queue, out_queue: queue.Queue, boost: float):
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break

            results = detector.detect_emotions(frame)

            if results:
                emotions = results[0]["emotions"]
                emotions["sad"] *= boost

                best_emotion = max(emotions, key=emotions.get)
                confidence = emotions[best_emotion]
                box = results[0]["box"]

                # Drain old result and push new one
                try:
                    out_queue.get_nowait()
                except queue.Empty:
                    pass
                out_queue.put_nowait({
                    "emotion": best_emotion,
                    "confidence": confidence,
                    "emotions": emotions,
                    "box": box
                })
        except queue.Empty:
            continue
        except Exception:
            continue

# -----------------------------------
# CAMERA SETUP
# -----------------------------------
@st.cache_resource
def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
    return cap

camera = open_camera()

# -----------------------------------
# BACKGROUND WORKER
# -----------------------------------
frame_queue: queue.Queue = queue.Queue(maxsize=1)

worker = threading.Thread(
    target=emotion_worker,
    args=(frame_queue, result_queue, sad_boost),
    daemon=True
)
worker.start()

# -----------------------------------
# PLACEHOLDERS
# -----------------------------------
frame_placeholder = st.empty()
emotion_placeholder = st.empty()

# Sidebar: pre-create metric placeholders once (avoids sidebar churn)
st.sidebar.subheader("Emotion Scores")
sidebar_slots = {
    emotion: st.sidebar.empty()
    for emotion in ["angry", "fear", "happy", "sad", "surprise", "neutral"]
}
fps_slot = st.sidebar.empty()

# -----------------------------------
# STATE
# -----------------------------------
frame_count = 0
last_result = {"emotion": "Neutral", "confidence": 0.0, "emotions": {}, "box": None}

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

    # Push frame to worker every N frames (non-blocking drop if busy)
    if frame_count % process_every_n == 0:
        small = cv2.resize(frame, (320, 240))  # Worker gets smaller frame = faster
        try:
            frame_queue.get_nowait()           # Drop stale unprocessed frame
        except queue.Empty:
            pass
        try:
            frame_queue.put_nowait(small)
        except queue.Full:
            pass

    # Pull latest result (non-blocking)
    try:
        last_result = result_queue.get_nowait()
    except queue.Empty:
        pass

    # Draw bounding box on display frame (full-res)
    display = frame.copy()
    if last_result.get("box") is not None:
        x, y, w, h = last_result["box"]
        # Scale box from 320x240 -> 640x480
        x, y, w, h = x * 2, y * 2, w * 2, h * 2
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 100), 2)

    # Draw label
    label = f"{last_result['emotion']}  {last_result['confidence']:.2f}"
    cv2.putText(display, label, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 100), 2, cv2.LINE_AA)

    # Convert BGR -> RGB for Streamlit
    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

    # Show frame
    frame_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

    # Update emotion label
    emotion_placeholder.markdown(
        f"## 🎭 **{last_result['emotion'].capitalize()}** "
        f"— confidence: `{last_result['confidence']:.2f}`"
    )

    # Update sidebar scores (pre-allocated slots, no rerenders)
    for emotion, slot in sidebar_slots.items():
        score = last_result["emotions"].get(emotion, 0.0)
        bar = "█" * int(score * 20)
        slot.markdown(f"`{emotion:<9}` {bar} `{score:.2f}`")

    # FPS
    fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
    fps_slot.markdown(f"**FPS:** `{fps:.1f}`")

# -----------------------------------
# CLEANUP
# -----------------------------------
frame_queue.put(None)  # Signal worker to stop
camera.release()
st.info("Camera stopped.")
