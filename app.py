from flask import Flask, render_template, Response, request, jsonify
import cv2
import joblib
import numpy as np
from skimage.feature import hog
import pygame
import threading
import time

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("fit_model/emotion_svm_model.pkl")
label_encoder = joblib.load("fit_model/label_encoder.pkl")

# Music control
pygame.mixer.init()
last_emotion = None
last_play_time = 0
muted = False

emotion_music = {
    "angry": "fit_model/songs/angry.mp3",
    "disgust": "fit_model/songs/disgust.mp3",
    "fear": "fit_model/songs/fear.mp3",
    "happy": "fit_model/songs/happy.mp3",
    "neutral": "fit_model/songs/neutral.mp3",
    "sad": "fit_model/songs/sad.mp3",
    "surprise": "fit_model/songs/surprise.mp3"
}

def play_music(emotion):
    global last_emotion, last_play_time, muted
    if muted:
        return

    current_time = time.time()
    if emotion != last_emotion or (current_time - last_play_time > 10):
        last_emotion = emotion
        last_play_time = current_time
        song_path = emotion_music.get(emotion)
        if song_path:
            def _play():
                pygame.mixer.music.stop()
                pygame.mixer.music.load(song_path)
                pygame.mixer.music.play()
            threading.Thread(target=_play, daemon=True).start()

# Webcam and detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
streaming = True

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        if not streaming:
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (48, 48))
            hog_feat = hog(roi_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            hog_feat = np.array(hog_feat, dtype='float32').reshape(1, -1)

            prediction = model.predict(hog_feat)
            emotion = label_encoder.inverse_transform(prediction)[0]
            play_music(emotion)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Web routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_stream', methods=['POST'])
def toggle_stream():
    global streaming
    streaming = not streaming
    return jsonify({'streaming': streaming})

@app.route('/toggle_mute', methods=['POST'])
def toggle_mute():
    global muted
    muted = not muted
    if muted:
        pygame.mixer.music.pause()
    else:
        pygame.mixer.music.unpause()
    return jsonify({'muted': muted})

if __name__ == "__main__":
    app.run(debug=True)
