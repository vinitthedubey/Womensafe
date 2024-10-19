'''from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# load model
model = load_model('gender_detection.h5')  # Ensure this matches the saved model file extension

# open webcam
webcam = cv2.VideoCapture(0)

classes = ['man', 'woman']

# loop through frames
while webcam.isOpened():
    # read frame from webcam 
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):
        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0]  # model.predict returns a 1D array

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("Gender Detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()'''


'''import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cvlib as cv
from datetime import datetime, timedelta

# Load gender detection model
model = load_model('gender_detection.h5')

# Open webcam
webcam = cv2.VideoCapture(0)

# Define classes and initialize variables
classes = ['man', 'woman']
gender_count = {'man': 0, 'woman': 0}
night_time_start = 18  # 6 PM
night_time_end = 6  # 6 AM
hotspot_log = []
sos_gesture_detected = False
gesture_counter = 0
sos_threshold = 5  # Threshold for rapid movement to be considered an SOS gesture
gesture_detection_interval = timedelta(seconds=2)  # Time interval for counting gestures
last_gesture_time = datetime.now()

# Function to check if it's nighttime
def is_night_time():
    current_hour = datetime.now().hour
    return current_hour >= night_time_start or current_hour < night_time_end

# Function to check if a woman is surrounded by men
def woman_surrounded(gender_count):
    return gender_count['woman'] > 0 and gender_count['man'] > 1

# Function to detect rapid hand movements for SOS
def detect_sos_gesture(frame):
    global gesture_counter, last_gesture_time
    current_time = datetime.now()

    # Detect hand motion (basic example using frame differencing)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if not hasattr(detect_sos_gesture, 'prev_frame'):
        detect_sos_gesture.prev_frame = gray
        return False

    frame_delta = cv2.absdiff(detect_sos_gesture.prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check for significant movement
    for contour in cnts:
        if cv2.contourArea(contour) > 5000:  # Adjust area threshold as needed
            gesture_counter += 1
            last_gesture_time = current_time
            if gesture_counter >= sos_threshold:
                sos_gesture_detected = True
                gesture_counter = 0  # Reset after detection
                return True

    # Reset gesture counter if no movement detected within the interval
    if current_time - last_gesture_time > gesture_detection_interval:
        gesture_counter = 0

    detect_sos_gesture.prev_frame = gray
    return False

# Function to identify hotspots
def identify_hotspot():
    global hotspot_log
    current_time = datetime.now()

    # Keep track of recent alerts (within the last 5 minutes)
    hotspot_log = [log for log in hotspot_log if current_time - log < timedelta(minutes=5)]

    # If more than 3 alerts in the last 5 minutes, consider it a hotspot
    if len(hotspot_log) >= 3:
        return True
    return False

# Loop through frames
while webcam.isOpened():
    # Read frame from webcam 
    status, frame = webcam.read()

    # Apply face detection
    faces, confidence = cv.detect_face(frame)

    # Reset gender count for each frame
    gender_count = {'man': 0, 'woman': 0}

    # Loop through detected faces
    for idx, f in enumerate(faces):
        # Get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # Preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on face
        conf = model.predict(face_crop)[0]  # model.predict returns a 1D array

        # Get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]
        gender_count[label] += 1

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Gender Distribution Analysis
    print(f"Gender Distribution - Men: {gender_count['man']}, Women: {gender_count['woman']}")

    # Identify a lone woman at night
    if is_night_time() and gender_count['woman'] == 1 and gender_count['man'] == 0:
        print("Alert: Lone woman detected at night!")
        cv2.putText(frame, "Alert: Lone woman detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        hotspot_log.append(datetime.now())  # Log alert for hotspot analysis

    # Detect if a woman is surrounded by men
    if woman_surrounded(gender_count):
        print("Alert: Woman surrounded by men!")
        cv2.putText(frame, "Alert: Woman surrounded by men!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        hotspot_log.append(datetime.now())  # Log alert for hotspot analysis

    # Recognize SOS gesture
    if detect_sos_gesture(frame):
        print("SOS Alert: Distress signal detected!")
        cv2.putText(frame, "SOS Alert: Distress signal detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        hotspot_log.append(datetime.now())  # Log alert for hotspot analysis

    # Identify hotspots
    if identify_hotspot():
        print("Alert: Hotspot detected!")
        cv2.putText(frame, "Alert: Hotspot detected!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Display output
    cv2.imshow("Women Safety Analytics", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()'''


import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cvlib as cv
from datetime import datetime, timedelta

# Load gender detection model
model = load_model('gender_detection.h5')

# Open webcam
webcam = cv2.VideoCapture(0)

# Define classes and initialize variables
classes = ['man', 'woman']
gender_count = {'man': 0, 'woman': 0}
night_time_start = 18  # 6 PM
night_time_end = 6  # 6 AM
hotspot_log = []
sos_gesture_detected = False
gesture_counter = 0
sos_threshold = 5  # Threshold for rapid movement to be considered an SOS gesture
gesture_detection_interval = timedelta(seconds=2)  # Time interval for counting gestures
last_gesture_time = datetime.now()

# Function to check if it's nighttime
def is_night_time():
    current_hour = datetime.now().hour
    return current_hour >= night_time_start or current_hour < night_time_end

# Function to check if a woman is surrounded by men
def woman_surrounded(gender_count):
    return gender_count['woman'] > 0 and gender_count['man'] > 1

# Function to detect rapid hand movements for SOS
def detect_sos_gesture(frame):
    global gesture_counter, last_gesture_time
    current_time = datetime.now()

    # Detect hand motion (basic example using frame differencing)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if not hasattr(detect_sos_gesture, 'prev_frame'):
        detect_sos_gesture.prev_frame = gray
        return False

    frame_delta = cv2.absdiff(detect_sos_gesture.prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check for significant movement
    for contour in cnts:
        if cv2.contourArea(contour) > 5000:  # Adjust area threshold as needed
            gesture_counter += 1
            last_gesture_time = current_time
            if gesture_counter >= sos_threshold:
                gesture_counter = 0  # Reset after detection
                return True

    # Reset gesture counter if no movement detected within the interval
    if current_time - last_gesture_time > gesture_detection_interval:
        gesture_counter = 0

    detect_sos_gesture.prev_frame = gray
    return False

# Function to identify hotspots
def identify_hotspot():
    global hotspot_log
    current_time = datetime.now()

    # Keep track of recent alerts (within the last 5 minutes)
    hotspot_log = [log for log in hotspot_log if current_time - log < timedelta(minutes=5)]

    # If more than 3 alerts in the last 5 minutes, consider it a hotspot
    return len(hotspot_log) >= 3

# Loop through frames
while webcam.isOpened():
    # Read frame from webcam 
    status, frame = webcam.read()

    # Apply face detection
    faces, confidence = cv.detect_face(frame)

    # Reset gender count for each frame
    gender_count = {'man': 0, 'woman': 0}

    # Loop through detected faces
    for idx, f in enumerate(faces):
        # Get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # Preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on face
        conf = model.predict(face_crop)[0]  # model.predict returns a 1D array

        # Get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]
        gender_count[label] += 1

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Gender Distribution Analysis
    print(f"Gender Distribution - Men: {gender_count['man']}, Women: {gender_count['woman']}")

    # Identify a lone woman at night
    if is_night_time() and gender_count['woman'] == 1 and gender_count['man'] == 0:
        print("Alert: Lone woman detected at night!")
        cv2.putText(frame, "Alert: Lone woman detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        hotspot_log.append(datetime.now())  # Log alert for hotspot analysis

    # Detect if a woman is surrounded by men
    if woman_surrounded(gender_count):
        print("Alert: Woman surrounded by men!")
        cv2.putText(frame, "Alert: Woman surrounded by men!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        hotspot_log.append(datetime.now())  # Log alert for hotspot analysis

    # Recognize SOS gesture
    if detect_sos_gesture(frame):
        print("SOS Alert: Distress signal detected!")
        cv2.putText(frame, "SOS Alert: Distress signal detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        hotspot_log.append(datetime.now())  # Log alert for hotspot analysis

    # Identify hotspots
    if identify_hotspot():
        print("Alert: Hotspot detected!")
        cv2.putText(frame, "Alert: Hotspot detected!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Display output
    cv2.imshow("Women Safety Analytics", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cvlib as cv
from datetime import datetime, timedelta

# Load gender detection model
model = load_model('gender_detection.h5')

# Open webcam
webcam = cv2.VideoCapture(0)

# Define classes and initialize variables
classes = ['man', 'woman']
gender_count = {'man': 0, 'woman': 0}
night_time_start = 18  # 6 PM
night_time_end = 6  # 6 AM
hotspot_log = []
sos_gesture_detected = False
gesture_counter = 0
sos_threshold = 5  # Threshold for rapid movement to be considered an SOS gesture
gesture_detection_interval = timedelta(seconds=2)  # Time interval for counting gestures
last_gesture_time = datetime.now()

# Function to check if it's nighttime
def is_night_time():
    current_hour = datetime.now().hour
    return current_hour >= night_time_start or current_hour < night_time_end

# Function to check if a woman is surrounded by men
def woman_surrounded(gender_count):
    return gender_count['woman'] > 0 and gender_count['man'] > 1

# Function to detect rapid hand movements for SOS
def detect_sos_gesture(frame):
    global gesture_counter, last_gesture_time
    current_time = datetime.now()

    # Detect hand motion (basic example using frame differencing)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if not hasattr(detect_sos_gesture, 'prev_frame'):
        detect_sos_gesture.prev_frame = gray
        return False

    frame_delta = cv2.absdiff(detect_sos_gesture.prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check for significant movement
    for contour in cnts:
        if cv2.contourArea(contour) > 5000:  # Adjust area threshold as needed
            gesture_counter += 1
            last_gesture_time = current_time
            if gesture_counter >= sos_threshold:
                gesture_counter = 0  # Reset after detection
                return True

    # Reset gesture counter if no movement detected within the interval
    if current_time - last_gesture_time > gesture_detection_interval:
        gesture_counter = 0

    detect_sos_gesture.prev_frame = gray
    return False

# Function to identify hotspots
def identify_hotspot():
    global hotspot_log
    current_time = datetime.now()

    # Keep track of recent alerts (within the last 5 minutes)
    hotspot_log = [log for log in hotspot_log if current_time - log < timedelta(minutes=5)]

    # If more than 3 alerts in the last 5 minutes, consider it a hotspot
    return len(hotspot_log) >= 3

# Loop through frames
while webcam.isOpened():
    # Read frame from webcam 
    status, frame = webcam.read()

    # Apply face detection
    faces, confidence = cv.detect_face(frame)

    # Reset gender count for each frame
    gender_count = {'man': 0, 'woman': 0}

    # Loop through detected faces
    for idx, f in enumerate(faces):
        # Get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # Preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on face
        conf = model.predict(face_crop)[0]  # model.predict returns a 1D array

        # Get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]
        gender_count[label] += 1

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Gender Distribution Analysis
    print(f"Gender Distribution - Men: {gender_count['man']}, Women: {gender_count['woman']}")

    # Identify a lone woman at night
    if is_night_time() and gender_count['woman'] == 1 and gender_count['man'] == 0:
        print("Alert: Lone woman detected at night!")
        cv2.putText(frame, "Alert: Lone woman detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        hotspot_log.append(datetime.now())  # Log alert for hotspot analysis

    # Detect if a woman is surrounded by men
    if woman_surrounded(gender_count):
        print("Alert: Woman surrounded by men!")
        cv2.putText(frame, "Alert: Woman surrounded by men!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        hotspot_log.append(datetime.now())  # Log alert for hotspot analysis

    # Recognize SOS gesture
    if detect_sos_gesture(frame):
        print("SOS Alert: Distress signal detected!")
        cv2.putText(frame, "SOS Alert: Distress signal detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        hotspot_log.append(datetime.now())  # Log alert for hotspot analysis

    # Identify hotspots
    if identify_hotspot():
        print("Alert: Hotspot detected!")
        cv2.putText(frame, "Alert: Hotspot detected!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Display output
    cv2.imshow("Women Safety Analytics", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
