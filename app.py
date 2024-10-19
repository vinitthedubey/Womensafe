from flask import Flask, render_template, request, jsonify, Response,redirect
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cvlib as cv
from datetime import datetime
import mediapipe as mp
from pymongo import MongoClient
from pymongo import UpdateOne
import plotly.express as px
import plotly.io as pio
import pandas as pd

# Load your CSV file
data = pd.read_csv('finalcensus.csv')


app = Flask(__name__)

# Load gender detection model
model = load_model('gender_detection_model_4.h5')

# MongoDB connection
client = MongoClient("mongodb+srv://vinit_dubey:1860Amul@cluster0.zjwfiov.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db2 = client.Women_Safety
incident_collection = db2.incident_report
safe_city_collection = db2.safe_city
collection5 = db2.appointment

def insert_or_update_incident_report(country, state, city):
    # Connection URL (Update this with your MongoDB connection string)
    mongo_uri = "mongodb+srv://vinit_dubey:1860Amul@cluster0.zjwfiov.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(mongo_uri)

    # Access the Women_Safety database and incident_report collection
    db = client['Women_Safety']
    collection = db['incident_report']

    # Define the filter for finding the document
    filter_criteria = {
        'country': country,
        'state': state,
        'city': city
    }

    # Define the update operation
    update_operation = {
        '$inc': {'incident': 1}  # Increment the incident count by 1
    }

    try:
        # Update the document if it exists, or insert a new one if it doesn't
        result = collection.update_one(
            filter_criteria,
            update_operation,
            upsert=True  # Create a new document if no match is found
        )
        
        if result.upserted_id:
            print(f"New record created with id: {result.upserted_id} and incident count set to 1.")
        else:
            print("Incident count incremented by 1 for the existing record.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the connection
        client.close()


def insert_or_update_highprob_incident_report(country, state, city):
    # Connection URL (Update this with your MongoDB connection string)
    mongo_uri = "mongodb+srv://vinit_dubey:1860Amul@cluster0.zjwfiov.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(mongo_uri)

    # Access the Women_Safety database and incident_report collection
    db = client['Women_Safety']
    collection = db['incident_report']

    # Define the filter for finding the document
    filter_criteria = {
        'country': country,
        'state': state,
        'city': city
    }

    # Define the update operation
    update_operation = {
        '$inc': {'high_prob_incident': 1}  # Increment the incident count by 1
    }

    try:
        # Update the document if it exists, or insert a new one if it doesn't
        result = collection.update_one(
            filter_criteria,
            update_operation,
            upsert=True  # Create a new document if no match is found
        )
        
        if result.upserted_id:
            print(f"New record created with id: {result.upserted_id} and incident count set to 1.")
        else:
            print("Incident count incremented by 1 for the existing record.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the connection
        client.close()


# Placeholder variables
men_count = 0
women_count = 0
sos_status = False
lone=False
incident = []
country,city,state=None,None,None
light_incident,strong_incident=False,False



# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def is_nighttime():
    """Check if the current time is between 6 PM and 5 AM."""
    current_hour = datetime.now().hour
    return current_hour >= 18 or current_hour <= 5  # Nighttime is 6 PM to 5 AM

def is_woman_lone_at_night(men_count, women_count):
    """Check if a woman is alone at night."""
    # If it's night and only one woman is detected with no men
    if women_count == 1 and men_count == 0 and is_nighttime():
        return True
    return False


def woman_surrounded(men_count, women_count):
    if ( (women_count!=0 and men_count!=0) and men_count>=3*women_count ):
        return True
    else:
        return False
    


def check_sos_gesture(frame):
    global hands

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        hand_positions = []  # Store wrist positions to detect crossed hands
        
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame (optional for visualization)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract key landmarks for gesture recognition
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
            middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
            ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
            pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]

            # Add wrist position for crossed hands detection
            hand_positions.append(wrist)

            # 1. Open Palm Gesture (all fingers spread)
            if (thumb_tip.y < thumb_ip.y and
                index_tip.y < index_dip.y and
                middle_tip.y < middle_dip.y and
                ring_tip.y < ring_dip.y and
                pinky_tip.y < pinky_dip.y):
                return "open_palm"

            # 2. Closed Fist Gesture (all fingers curled into palm)
            if (thumb_tip.y > thumb_ip.y and
                index_tip.y > index_dip.y and
                middle_tip.y > middle_dip.y and
                ring_tip.y > ring_dip.y and
                pinky_tip.y > pinky_dip.y):
                return "closed_fist"

            # 3. Thumbs Up Gesture (thumb extended up, others curled)
            if (thumb_tip.y < thumb_ip.y and
                index_tip.y > index_dip.y and
                middle_tip.y > middle_dip.y and
                ring_tip.y > ring_dip.y and
                pinky_tip.y > pinky_dip.y):
                return "thumbs_up"

            # 4. V-sign Gesture (index and middle extended, others curled)
            if (index_tip.y < index_dip.y and
                middle_tip.y < middle_dip.y and
                ring_tip.y > ring_dip.y and
                pinky_tip.y > pinky_dip.y):
                return "v_sign"
        
        # 5. Crossed Hands Gesture (both hands crossing)
        if len(hand_positions) == 2:  # Ensure there are two hands detected
            wrist_1, wrist_2 = hand_positions[0], hand_positions[1]

            # Check if the x-coordinates of the wrists are overlapping
            if abs(wrist_1.x - wrist_2.x) < 0.05:  # Adjust threshold as needed for sensitivity
                return "crossed_hands"

    return None

# Function to detect SOS gesture
def detect_sos_gesture(frame):
    global sos_status
    gesture = check_sos_gesture(frame)

    # Set sos_status to True if any safety gesture is detected
    if gesture in ["open_palm", "closed_fist", "thumbs_up", "v_sign", "crossed_hands"]: 
        sos_status = True
    else:
        sos_status = False


# Function to update detection info
def update_detection(frame):
    global men_count, women_count, sos_status, incident,lone,light_incident,strong_incident

    men_count = 0
    women_count = 0
    lone=False
    light_incident = False
    strong_incident = False

    # Face detection
    faces, confidence = cv.detect_face(frame)

    for face in faces:
        (startX, startY, endX, endY) = face
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Predict gender
        conf = model.predict(face_crop)[0]
        label = np.argmax(conf)

        if label == 0:  # man
            men_count += 1
        else:  # woman
            women_count += 1
    if is_woman_lone_at_night(men_count, women_count):
        lone=True

    # Detect SOS gesture
    detect_sos_gesture(frame)  # This directly updates sos_status

    # If a woman is surrounded by men or SOS detected, log it as a hotspot
    if woman_surrounded(men_count, women_count) :
        strong_incident=True
        insert_or_update_highprob_incident_report(country,state,city)
        insert_or_update_incident_report(country,state,city)
    elif  (sos_status):
        light_incident=True
        incident.append(f"Hotspot detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        insert_or_update_incident_report(country,state,city)
    

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/index.html')
def indexsite():
    return render_template('index.html')

@app.route('/location.html', methods=['GET'])
def register():
    return render_template('location.html')

@app.route('/location', methods=['POST'])
def location():
    global country,state,city
    country = request.form.get('country')
    state = request.form.get('state')
    city = request.form.get('city')
    print(country, " - ", state, " - ", city)
    return render_template('main.html')

@app.route('/getDetectionInfo')
def get_detection_info():
    global men_count, women_count, sos_status, incident
    data = {
        'menCount': men_count,
        'womenCount': women_count,
        'sosStatus': sos_status,
        'hotspots': incident,
        'womenSurrounded': woman_surrounded(men_count, women_count),
        'currentTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'womenlone':lone,
        'light_incident':light_incident,
        'strong_incident':strong_incident,
        'city':city
        
    }
    return jsonify(data)

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Update detection info
        update_detection(frame)

        # Encode the frame into a JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Use frame as response
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Routing websites start


@app.route('/admin')
def index():
    return render_template('admin.html')

@app.route('/admin.html')
def admin():
    return render_template('admin.html')


@app.route('/index1.html')
def hiome():
    return render_template('index1.html')

@app.route('/about.html')
def hidome():
    return render_template('about.html')

@app.route('/appointment.html')
def hifome():
    return render_template('appointment.html')

@app.route('/contact.html')
def hiofme():
    return render_template('contact.html')

@app.route('/feature.html')
def hiovme():
    return render_template('feature.html')

@app.route('/service.html')
def hiosme():
    return render_template('service.html')

@app.route('/team.html')
def hiomtre():
    return render_template('team.html')

@app.route('/testimonial.html')
def hiomete():
    return render_template('testimonial.html')



#Routing websites end





# Route to handle form submission and save data to MongoDB
@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    name = request.form['name']
    email = request.form['email']
    mobile = request.form['mobile']
    officer = request.form['officer']
    date = request.form['date']
    time = request.form['time']
    problem = request.form['problem']

    appointment_data = {
        'name': name,
        'email': email,
        'mobile': mobile,
        'officer': officer,
        'date': date,
        'time': time,
        'problem': problem
    }

    collection5.insert_one(appointment_data)
    return render_template('success.html', name=name)


# Route to display the report form
@app.route('/appointment_status')
def appointment_status():
    return render_template('appointment_status.html')

# Route to generate the report based on the name entered
@app.route('/report', methods=['POST'])
def report():
    name = request.form['name']
    report_data = collection5.find_one({'name': name})
    
    if report_data:
        return render_template('report.html', data=report_data)
    else:
        return "No appointment found for the entered name."





@app.route('/submit', methods=['POST'])
def submit():
    data12 = request.json  # Expecting JSON
    country = data12.get('country')
    state = data12.get('state')
    city = data12.get('city')
    app.logger.info(f"Received: Country={country}, State={state}, City={city}")

    # Check CSV for city data
    city_data = data[(data['Name'].str.lower().str.strip() == city.lower().strip()) & (data['Total/\nRural/\nUrban'].str.lower().str.strip() == 'Total'.lower().strip())]
    
    if not city_data.empty:
        total_male = city_data['Male'].values[0]
        total_female = city_data['Female'].values[0]
        
        # Get incident and report values from MongoDB
        incidents = list(incident_collection.find({"country": country, "state": state, "city": city}, {"incident": 1, "report": 1}))
        
        incident_count = len(incidents)
        report_count = sum(1 for inc in incidents if 'report' in inc)

        # Create Plotly figures
        pie_fig = px.pie(names=['Male', 'Female'], values=[total_male, total_female], color_discrete_sequence=['blue', 'pink'])
        bar_fig_incidents = px.bar(x=[city], y=[incident_count], title='Number of Incidents')
        bar_fig_reports = px.bar(x=[city], y=[report_count], title='Number of Reports')

        return jsonify({
            'pie': pie_fig.to_json(),
            'bar_incidents': bar_fig_incidents.to_json(),
            'bar_reports': bar_fig_reports.to_json(),
        })
    else:
        return jsonify({'error': 'City not found in data.'}), 404


@app.route('/getCityData', methods=['POST'])
def get_city_data():
    selected_city = city

    # Filter the data for the selected city and "Total" area type
    city_data = data[(data['Name'].str.lower().str.strip() == city.lower().strip()) & (data['Total/\nRural/\nUrban'].str.lower().str.strip() == 'Total'.lower().strip())]
    if not city_data.empty:
        male_population = city_data['Male'].values[0]
        female_population = city_data['Female'].values[0]

        # Create a pie chart using Plotly
        fig = px.pie(
            values=[male_population, female_population],
            names=['Males', 'Females'],
            title=f'Gender Distribution in {selected_city}'
        )

        # Convert Plotly figure to JSON to be used in the HTML template
        graph_json = pio.to_json(fig)
        return jsonify(graph_json)
    else:
        return jsonify({'error': 'City not found or no Total data available'}), 404


@app.route('/update_safe', methods=['POST'])
def update_safe():
    country1 = request.form.get('country')
    state1 = request.form.get('state')
    city1 = request.form.get('city')
    safe_status = request.form.get('safe')

    if safe_status == 'True':
        safe_city_collection.update_one({"country": country1, "state": state1, "city": city1}, {"$set": {"safe": True}}, upsert=True)
        
    else:
        safe_city_collection.update_one({"country": country1, "state": state1, "city": city1}, {"$set": {"safe": False}}, upsert=True)

    return jsonify({"status": "updated"})

if __name__ == "__main__":
    app.run(debug=True) 