from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from keras.models import model_from_json
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
json_file = open('new_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("final_model_weights.h5")

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

last_emotion = None

# OMDB API endpoint
api_key = os.environ.get('API_KEY')

# Spotify API credentials
client_id = os.environ.get('CLIENT_ID')
client_secret = os.environ.get('CLIENT_SECRET')

# Authenticate with Spotify API
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# Function to generate frames for the video feed
cap = cv2.VideoCapture(0)

def search_movies_by_mood(mood):
    url = f'http://www.omdbapi.com/?apikey={api_key}&s={mood}&type=movie'
    response = requests.get(url)
    data = response.json()
    return data['Search']

def generate_frames():
    global last_emotion
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            roi_color_frame = frame[y:y + h, x:x + w]
            resized_img = cv2.resize(roi_color_frame, (224, 224))
            resized_img = np.expand_dims(resized_img, axis=0)
            
            emotion_prediction = emotion_model.predict(resized_img)
            maxindex = int(np.argmax(emotion_prediction))
            last_emotion = emotion_dict[maxindex]
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()


# Route for the main page
@app.route('/')
def index():
    return render_template('home.html')

# Route for the page displaying the camera feed
@app.route('/detect_mood')
def detect_mood():
    return render_template('detect_mood.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_emotion')
def capture_emotion():
    global last_emotion
    # print("Last Emotion:", last_emotion)
    return jsonify({"emotion": last_emotion})

@app.route('/result')
def result():
    # last_emotion = request.args.get('last_emotion')
    global last_emotion
    # print(last_emotion)
    return render_template('result.html', last_emotion=last_emotion)

@app.route('/get_songs_and_movies')
def get_songs_and_movies():
    global last_emotion
    results_hindi = sp.search(q=f'mood:{last_emotion} lang:hi', type='track', limit=5)
    results_english = sp.search(q=f'mood:{last_emotion} lang:en', type='track', limit=5)
    
    # Combine the results
    combined_results = results_hindi['tracks']['items'] + results_english['tracks']['items']
    
    songs = []
    for track in combined_results:
        songs.append(f"{track['name']} - {', '.join([artist['name'] for artist in track['artists']])}")

    movies = []
    movies_data = search_movies_by_mood(last_emotion)
    for movie in movies_data:
        movies.append(f"{movie['Title']} ({movie['Year']})")

    return jsonify({"songs": songs, "movies": movies})

if __name__ == '__main__':
    app.run(debug=True)
