import cv2
import joblib
import numpy as np
from skimage.feature import hog
import pygame
import threading
import time

pygame.mixer.init()
last_emotion = None
last_play_time = 0

emotion_music = {
    "angry": "songs/angry.mp3",
    "disgust": "songs/disgust.mp3",
    "fear": "songs/fear.mp3",
    "happy": "songs/happy.mp3",
    "neutral": "songs/neutral.mp3",
    "sad": "songs/sad.mp3",
    "surprise": "songs/surprise.mp3"
}


def play_music(emotion):
    global last_emotion, last_play_time
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

# Load model and label encoder
model = joblib.load("emotion_svm_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")



# Start webcam
cap = cv2.VideoCapture(0)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
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

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🛑 Interrupted manually.")

finally:
    cap.release()
    cv2.destroyAllWindows()
