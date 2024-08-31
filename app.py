from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the pre-trained gender detection model
model = load_model('gender_detection.h5')  # Replace with your actual model path

# Load the face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define labels
labels = ['Male', 'Female']

def generate_frames():
    camera = cv2.VideoCapture(0)  # Open the webcam
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert the frame to grayscale (for face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extract the face region
            face_crop = frame[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (96, 96))  # Resize to the model's input size
            face_crop = face_crop / 255.0  # Normalize pixel values
            face_crop = np.expand_dims(face_crop, axis=0)  # Reshape for the model

            # Predict gender
            gender_prediction = model.predict(face_crop)
            gender_label = labels[np.argmax(gender_prediction)]

            # Display the label on the video frame
            cv2.putText(frame, gender_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Encode the frame into a byte format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()  # Release the camera when done

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
