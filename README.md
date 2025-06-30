
# 🎭 Real-Time Emotion Music Player 🎵

This Flask-based application detects real-time facial emotions using a webcam, plays corresponding music, and visualizes emotion history using a live chart.

---

## 🚀 Features

- 🎥 Live Webcam Feed with Face Detection
- 🧠 Emotion Recognition (using SVM + HOG)
- 🎶 Music Playback Based on Detected Emotion
- 🔇 Mute/Unmute Music
- 🎛️ Start/Stop Webcam Stream
- 📊 Real-Time Emotion Bar Chart
- 🧊 Loading Spinner During Initialization

---

## 📂 Project Structure

```
emotion_app/
├── app.py                       # Flask backend
├── fit_model/
│   ├── emotion_svm_model.pkl    # Trained SVM model
│   ├── label_encoder.pkl        # Label encoder for emotion labels
│   └── songs/                   # Folder with emotion-mapped MP3 files
│       ├── happy.mp3
│       ├── sad.mp3
│       └── ...
├── templates/
│   └── index.html               # Web interface
├── static/
│   └── (optional) stylesheets
└── README.md
```

---

## 🛠️ Requirements

- Python 3.9 or higher
- OpenCV
- Flask
- pygame
- joblib
- scikit-learn
- scikit-image

Install using:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
python app.py
```

Then open your browser at:

```
http://127.0.0.1:5000
```

---

## ⚙️ Controls

- 🟢 **Stop Webcam** / 🔴 **Start Webcam**
- 🔊 **Mute** / 🔇 **Unmute**
- 📈 Emotion bar chart updates every 2 seconds

---

## 📦 Model

The emotion recognition model is based on:

- **HOG features** extracted from 48x48 grayscale faces
- **LinearSVC** trained on the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 📌 Notes

- Place MP3 files for each emotion inside the `fit_model/songs/` folder
- Ensure your webcam is functional
- Chrome or Edge recommended for webcam access

---

## 👨‍💻 Made by Allen with Flask + OpenCV + Machine Learning 🎧
