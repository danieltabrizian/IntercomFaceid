import os
import paho.mqtt.client as mqtt
import time
import json
import cv2
import numpy as np
import insightface

# MQTT broker address and credentials
mqtt_broker = os.getenv("MQTT_BROKER", "core-mosquitto")
mqtt_port = 1883
mqtt_username = os.getenv("MQTT_USERNAME", "mqtt")
mqtt_password = os.getenv("MQTT_PASSWORD", "mqtt")

# File to store face embeddings and names
FACE_DATA_FILE = "faces_data.json"

# Initialize MQTT client
mqtt_client = mqtt.Client()

if mqtt_username and mqtt_password:
    mqtt_client.username_pw_set(mqtt_username, mqtt_password)

mqtt_client.connect(mqtt_broker, mqtt_port, 60)

# Global states for face data
known_face_encodings = []
known_face_names = []

# MQTT topics for states and commands
bell_state_topic = "homeassistant/binary_sensor/bell_run/state"
learn_face_command_topic = "homeassistant/button/learn_new_face/set"
unlock_door_command_topic = "homeassistant/button/unlock_door/set"

# Initialize InsightFace model
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0)

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

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

# Function to generate face embedding using InsightFace
def get_face_embedding(image):
    faces = model.get(image)
    if len(faces) > 0:
        embedding = faces[0].embedding
        return np.array(embedding)
    else:
        raise ValueError("No face detected")

# Function to capture face embeddings for 60 seconds and store them
def learn_new_face():
    print("Learning new face...")
    video_capture = cv2.VideoCapture(0)
    start_time = time.time()
    while time.time() - start_time < 60:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video")
            break
        try:
            embedding = get_face_embedding(frame)
            known_face_encodings.append(embedding)
            known_face_names.append("New Face")
            print("Face embedding captured!")
        except Exception as e:
            print(f"Error generating embedding: {e}")
    save_face_data()
    video_capture.release()
    cv2.destroyAllWindows()
    print("New face learned!")

# Function to compare detected face with known faces
def ring_bell():
    print("Bell rung!")
    publish_bell_state("ON")
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture video")
    else:
        try:
            embedding = get_face_embedding(frame)
            similarities = [cosine_similarity(embedding, known_face) for known_face in known_face_encodings]
            best_match_index = np.argmax(similarities)
            if similarities[best_match_index] > 0.7:
                print(f"Face recognized with {similarities[best_match_index] * 100:.2f}% similarity! Unlocking door...")
                unlock_door()
            else:
                print(f"Unknown face with {similarities[best_match_index] * 100:.2f}% similarity. Access denied.")
        except Exception as e:
            print(f"Error during face comparison: {e}")
    video_capture.release()
    cv2.destroyAllWindows()
    publish_bell_state("OFF")

# Function to publish bell state
def publish_bell_state(state):
    mqtt_client.publish(bell_state_topic, state)
    print(f"Published bell state: {state}")

# Function to simulate unlocking the door
def unlock_door():
    print("Unlocking the door...")
    time.sleep(2)
    print("Door unlocked!")

# Function to handle MQTT messages
def on_message(client, userdata, msg):
    if msg.topic == learn_face_command_topic:
        learn_new_face()
    elif msg.topic == unlock_door_command_topic:
        unlock_door()

# Main MQTT setup
def main():
    load_face_data()
    mqtt_client.subscribe(learn_face_command_topic)
    mqtt_client.subscribe(unlock_door_command_topic)
    mqtt_client.on_message = on_message
    mqtt_client.loop_start()
    while True:
        ring_bell()
        time.sleep(30)

if __name__ == "__main__":
    main()