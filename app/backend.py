"""
backend.py
----------
All camera and emotion detection logic.
"""

import cv2
import queue
import threading
from fer import FER


# -----------------------------------
# CONSTANTS
# -----------------------------------
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
WORKER_WIDTH = 320
WORKER_HEIGHT = 240
SAD_BOOST = 1.20
EMOTIONS = ["angry", "fear", "happy", "sad", "surprise", "neutral"]


# -----------------------------------
# DETECTOR
# -----------------------------------
def load_detector() -> FER:
    """Load FER model. Call once and reuse."""
    return FER(mtcnn=False)  # Haar cascade — fast


# -----------------------------------
# CAMERA
# -----------------------------------
def open_camera(index: int = 0) -> cv2.VideoCapture:
    """Open webcam with optimized settings."""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Prevent frame buildup lag
    return cap


# -----------------------------------
# EMOTION WORKER (background thread)
# -----------------------------------
def emotion_worker(
    detector: FER,
    frame_queue: queue.Queue,
    result_queue: queue.Queue,
    sad_boost: float = SAD_BOOST,
) -> None:
    """
    Runs in a daemon thread.
    Reads frames from frame_queue, runs inference,
    pushes results to result_queue.
    """
    while True:
        try:
            frame = frame_queue.get(timeout=1)

            if frame is None:  # Shutdown signal
                break

            results = detector.detect_emotions(frame)

            if not results:
                continue

            emotions: dict = results[0]["emotions"]
            emotions["sad"] = emotions.get("sad", 0.0) * sad_boost

            best_emotion = max(emotions, key=emotions.get)
            confidence = emotions[best_emotion]
            box = results[0]["box"]

            payload = {
                "emotion": best_emotion,
                "confidence": confidence,
                "emotions": emotions,
                "box": box,
            }

            # Always keep only the latest result
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
            result_queue.put_nowait(payload)

        except queue.Empty:
            continue
        except Exception:
            continue


# -----------------------------------
# FRAME PROCESSOR
# -----------------------------------
def process_frame(frame, last_result: dict) -> tuple:
    """
    Draw bounding box and emotion label on frame.
    Returns (display_rgb_frame, fps) — ready for Streamlit.
    """
    display = frame.copy()

    box = last_result.get("box")
    if box is not None:
        x, y, w, h = box
        # Scale from worker resolution (320x240) to display (640x480)
        x, y, w, h = x * 2, y * 2, w * 2, h * 2
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 100), 2)

    label = f"{last_result['emotion']}  {last_result['confidence']:.2f}"
    cv2.putText(
        display, label, (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1,
        (0, 255, 100), 2, cv2.LINE_AA
    )

    # BGR -> RGB for Streamlit
    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    return display_rgb


# -----------------------------------
# WORKER LAUNCHER
# -----------------------------------
def start_worker(
    detector: FER,
    frame_queue: queue.Queue,
    result_queue: queue.Queue,
    sad_boost: float = SAD_BOOST,
) -> threading.Thread:
    """Spawn and return the background inference thread."""
    t = threading.Thread(
        target=emotion_worker,
        args=(detector, frame_queue, result_queue, sad_boost),
        daemon=True,
    )
    t.start()
    return t


# -----------------------------------
# FRAME PUSH HELPER
# -----------------------------------
def push_frame(frame_queue: queue.Queue, frame) -> None:
    """
    Resize frame and push to worker queue (non-blocking).
    Drops stale frame if worker is still busy.
    """
    small = cv2.resize(frame, (WORKER_WIDTH, WORKER_HEIGHT))
    try:
        frame_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        frame_queue.put_nowait(small)
    except queue.Full:
        pass


# -----------------------------------
# RESULT PULL HELPER
# -----------------------------------
def pull_result(result_queue: queue.Queue, last_result: dict) -> dict:
    """Pull latest result from queue, or return last known result."""
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return last_result
