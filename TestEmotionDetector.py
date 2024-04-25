import cv2
import numpy as np
from keras.models import model_from_json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests

# OMDB API endpoint
api_key = '8c22c924'

def search_movies_by_mood(mood):
    url = f'http://www.omdbapi.com/?apikey={api_key}&s={mood}&type=movie'
    response = requests.get(url)
    data = response.json()
    return data['Search']

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Spotify API credentials
client_id = 'bab87605921f4420a75be2f02b7cfa3c'
client_secret = '0ca7503815a645ff86c5034ba63414ca'

# Authenticate with Spotify API
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# load json and create model
# json_file = open('model/emotion_model.json', 'r')
json_file = open('new_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
# emotion_model.load_weights("model/emotion_model.h5")
emotion_model.load_weights("final_model_weights.h5")
print("Loaded model from disk")

# start the webcam feed
#cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
cap = cv2.VideoCapture(0)

last_emotion = None

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        #cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        roi_color_frame = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(roi_color_frame, (224, 224))
        resized_img = np.expand_dims(resized_img, axis=0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(resized_img)
        maxindex = int(np.argmax(emotion_prediction))
        last_emotion = emotion_dict[maxindex]
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


if last_emotion is not None:
    print("Last Emotion Detected:", last_emotion)
    # You need to define your logic to search for songs based on the detected mood here
    # For example:
    # Modify the search query to include both Hindi and English tracks
    results_hindi = sp.search(q=f'mood:{last_emotion} lang:hi', type='track', limit=5)
    results_english = sp.search(q=f'mood:{last_emotion} lang:en', type='track', limit=5)
    
    # Combine the results
    combined_results = results_hindi['tracks']['items'] + results_english['tracks']['items']
    
    print("Suggested songs\n")
    for track in combined_results:
        print(f"{track['name']} - {', '.join([artist['name'] for artist in track['artists']])}")

    movies = search_movies_by_mood(last_emotion)
    print("Suggested movies\n")
    for movie in movies:
        print(f"{movie['Title']} ({movie['Year']})")
else:
    print("No emotion detected.")