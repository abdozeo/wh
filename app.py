from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import face_recognition
import os
import base64
from io import BytesIO
from PIL import Image
import pickle  # Added for pickle support

app = Flask(__name__)

path = 'persons'
images = []
classNames = []
personsList = os.listdir(path)

for cl in personsList:
    curPersonn = cv2.imread(f'{path}/{cl}')
    images.append(curPersonn)
    classNames.append(os.path.splitext(cl)[0])

def findEncodeings(image):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Load face encodings from the pickle file
def loadFaceEncodings():
    with open('face_encodings.pickle', 'rb') as file:
        encodeListKnown = pickle.load(file)
    return encodeListKnown

encodeListKnown = loadFaceEncodings()
print('Face Encodings Loaded.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    image_data = data.get('image')

    if image_data:
        # Convert base64 image data to OpenCV image
        image = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
        img_np = np.array(image)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # Face recognition logic
        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        result = []

        for encodeface, faceLoc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(encodeListKnown, encodeface)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                result.append({'name': name})

        return jsonify(result)

    return jsonify({'error': 'No image data provided'})

if __name__ == '__main__':
    app.run(debug=True)