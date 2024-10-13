import os
import paho.mqtt.client as mqtt
import time
import json
import cv2
import face_recognition
import numpy as np

# MQTT broker address and credentials
mqtt_broker = os.getenv("MQTT_BROKER", "core-mosquitto")  # Use 'core-mosquitto' inside Home Assistant
mqtt_port = 1883  # Default MQTT port
mqtt_username = os.getenv("MQTT_USERNAME", "mqtt")
mqtt_password = os.getenv("MQTT_PASSWORD", "mqtt")

# File to store face embeddings and names
FACE_DATA_FILE = "faces_data.json"

# Initialize MQTT client
mqtt_client = mqtt.Client()

# Set MQTT credentials if needed
if mqtt_username and mqtt_password:
    mqtt_client.username_pw_set(mqtt_username, mqtt_password)

# Connect to the MQTT broker
mqtt_client.connect(mqtt_broker, mqtt_port, 60)

# Global states for face data
known_face_encodings = []
known_face_names = []

# MQTT topics for states and commands
bell_state_topic = "homeassistant/binary_sensor/bell_run/state"
learn_face_command_topic = "homeassistant/button/learn_new_face/set"
unlock_door_command_topic = "homeassistant/button/unlock_door/set"

# Function to save face data to a file
def save_face_data():
    face_data = {
        "encodings": [face.tolist() for face in known_face_encodings],
        "names": known_face_names
    }
    with open(FACE_DATA_FILE, 'w') as f:
        json.dump(face_data, f)
    print(f"Saved {len(known_face_encodings)} faces to {FACE_DATA_FILE}")

# Function to load face data from a file
def load_face_data():
    global known_face_encodings, known_face_names
    if os.path.exists(FACE_DATA_FILE):
        with open(FACE_DATA_FILE, 'r') as f:
            face_data = json.load(f)
            known_face_encodings = [np.array(encoding) for encoding in face_data["encodings"]]
            known_face_names = face_data["names"]
        print(f"Loaded {len(known_face_encodings)} faces from {FACE_DATA_FILE}")
    else:
        print("No face data file found, starting with an empty face database.")

# Function to capture face embeddings for 10 seconds and store them
def learn_new_face():
    print("Learning new face...")

    # Open the video stream (from /dev/video0)
    video_capture = cv2.VideoCapture(0)
    
    start_time = time.time()
    while time.time() - start_time < 10:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video")
            break

        # Detect faces in the frame
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Store the first face detected in the video
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append("New Face")  # You can customize the name or label here
            print("Face encoding captured!")

    # Save face data after learning
    save_face_data()

    # Release the video stream
    video_capture.release()
    cv2.destroyAllWindows()
    print("New face learned!")

# Function to compare detected face with known faces (during bell ring)
def ring_bell():
    global bell_running
    print("Bell rung!")
    bell_running = True
    publish_bell_state("ON")

    # Open the video stream and capture frames for face detection
    video_capture = cv2.VideoCapture(0)
    
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture video")
    else:
        # Detect faces in the frame
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Compare detected faces with known faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            # If a match is found, unlock the door
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                print("Face recognized! Unlocking door...")
                unlock_door()

    # Release the video stream
    video_capture.release()
    cv2.destroyAllWindows()

    bell_running = False
    publish_bell_state("OFF")

# Function to publish the state of the bell sensor
def publish_bell_state(state):
    mqtt_client.publish(bell_state_topic, state)
    print(f"Published bell state: {state}")

# Function to simulate unlocking the door
def unlock_door():
    print("Unlocking the door...")
    # Add your actual unlocking logic here
    time.sleep(2)
    print("Door unlocked!")

# Function to handle incoming messages (button presses)
def on_message(client, userdata, msg):
    if msg.topic == learn_face_command_topic:
        learn_new_face()

    elif msg.topic == unlock_door_command_topic:
        unlock_door()

# MQTT setup and main function
def main():
    # Load saved face data (if available)
    load_face_data()

    # Subscribe to command topics for buttons
    mqtt_client.subscribe(learn_face_command_topic)
    mqtt_client.subscribe(unlock_door_command_topic)

    # Set the callback function for when messages are received
    mqtt_client.on_message = on_message

    # Start MQTT client loop in a background thread
    mqtt_client.loop_start()

    # Main loop simulating the bell ringing every 30 seconds
    while True:
        ring_bell()
        time.sleep(30)  # Wait 30 seconds before ringing the bell again

if __name__ == "__main__":
    main()