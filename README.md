# 😊 Real-Time Emotion Recognition

A real-time facial emotion detection app built with Python and Streamlit.

## Demo
Detects 6 emotions live from your webcam:
`angry` `fear` `happy` `sad` `surprise` `neutral`

## Tech Stack
- [Streamlit](https://streamlit.io/) — web UI
- [FER](https://github.com/justinshenk/fer) — emotion detection (Mini-Xception model)
- [OpenCV](https://opencv.org/) — webcam capture
- [NumPy](https://numpy.org/) — array processing

## Installation

```bash
# Clone the repo
  git clone https://github.com/Prithwi035/emotion-recognition.git
cd emotion-recognition

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run emotion_app.py
```

## Features
- Live webcam feed with bounding box around detected face
- Real-time emotion label and confidence score
- Emotion scores displayed in sidebar
- Background thread for inference — no camera lag
- Adjustable frame processing rate via sidebar slider

## Project Structure
```
emotion-recognition/
├── emotion_app.py       # Main app
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Requirements
- Python 3.8+
- Webcam
