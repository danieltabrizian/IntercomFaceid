import cv2
import numpy as np
import insightface
import json
import os
import time
import threading
from datetime import datetime

class FaceRecognizer:
    def __init__(self):

        self.FACE_DATA_FILE = "faces_data.json"
        self.known_face_encodings = []
        self.known_face_names = []
        self.model = insightface.app.FaceAnalysis()
        self.model.prepare(ctx_id=0)
        self.load_face_data()

    def set_arduino(self,arduino):
        self.arduino = arduino
    def set_mqtt_client(self,mqtt_client):
        self.mqtt_client = mqtt_client

    def cosine_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def save_face_data(self):
        face_data = {
            "encodings": [[face.tolist() for face in faces] for faces in self.known_face_encodings],  # Save as lists of embeddings per person
            "names": self.known_face_names
        }
        with open(self.FACE_DATA_FILE, 'w') as f:
            json.dump(face_data, f)
        print(f"Saved face data to {self.FACE_DATA_FILE}")

    def load_face_data(self):
        if os.path.exists(self.FACE_DATA_FILE):
            with open(self.FACE_DATA_FILE, 'r') as f:
                face_data = json.load(f)
                # Load each list of embeddings per person
                self.known_face_encodings = [[np.array(face) for face in faces] for faces in face_data["encodings"]]
                self.known_face_names = face_data["names"]
            print(f"Loaded face data from {self.FACE_DATA_FILE}")
        else:
            print("No face data file found, starting with an empty face database.")

    def get_face_embedding(self, image):
        faces = self.model.get(image)
        if len(faces) > 0:
            embedding = faces[0].embedding
            return np.array(embedding)
        else:
            raise ValueError("No face detected")

    def learn_new_face(self, person_name=None):
        if person_name is None:
            # Use current date and time as the default name
            person_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        print(f"Learning new face for {person_name}...")
        video_capture = cv2.VideoCapture(0)
        start_time = time.time()
        session_embeddings = []  # Collect embeddings for the current face session

        while time.time() - start_time < 10:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture video")
                break

            try:
                # Get the embedding for the current frame (face)
                embedding = self.get_face_embedding(frame)

                if embedding is None:
                    print("No face detected. Skipping frame...")
                    continue

                # Check if the current face embedding matches any known face with > 70% similarity
                is_known_face = False
                for i, known_faces in enumerate(self.known_face_encodings):
                    for known_face in known_faces:
                        similarity = self.cosine_similarity(embedding, known_face)
                        if similarity > 0.7:
                            print(f"Embedding matches known person {self.known_face_names[i]} with {similarity * 100:.2f}% similarity. Skipping this embedding.")
                            is_known_face = True
                            break  # Stop checking other known faces if match found
                    if is_known_face:
                        break  # Stop checking other people if match found

                # If no match is found, add the embedding to the session collection
                if not is_known_face:
                # Compare the new embedding with the already collected session embeddings
                    is_similar_to_session_embedding = False
                    for session_embedding in session_embeddings:
                        session_similarity = self.cosine_similarity(embedding, session_embedding)
                        if session_similarity > 0.7:
                            print(f"New embedding is too similar to another embedding in this session ({session_similarity * 100:.2f}% similarity). Skipping this frame.")
                            is_similar_to_session_embedding = True
                            break  # Skip if it's similar to any session embedding

                    if not is_similar_to_session_embedding:
                        session_embeddings.append(embedding)
                        print(f"Collected new embedding for {person_name}.")
                    else:
                        print("Skipping this frame due to similarity with other session embeddings.")

            except Exception as e:
                print(f"Error generating embedding: {e}")

        # After the session, check if we collected any embeddings
        if session_embeddings:
            # If we have collected embeddings, save them as a new person
            self.known_face_encodings.append(session_embeddings)
            self.known_face_names.append(person_name)
            print(f"New face {person_name} learned and saved with {len(session_embeddings)} embeddings!")
        else:
            print(f"No unique embeddings collected for {person_name}, face might already exist.")

        # Save the updated face data
        self.save_face_data()
        if self.arduino is not None:
            self.arduino.unlock()
        video_capture.release()
        cv2.destroyAllWindows()
        print(f"Finished learning new face for {person_name}!")
        
    def captureFace(self, capture_time=30):
        def capture_video():
            video_capture = cv2.VideoCapture(0)
            start_time = time.time()

            while self.running:
                ret, frame = video_capture.read()
                if not ret:
                    print("Failed to capture video")
                    break

                try:
                    embedding = self.get_face_embedding(frame)
                    if embedding is None:
                        continue  # Skip to the next frame if no face is detected

                    similarities = []
                    for i, known_faces in enumerate(self.known_face_encodings):
                        face_similarities = [self.cosine_similarity(embedding, known_face) for known_face in known_faces]
                        similarities.append(np.max(face_similarities))  # Take the maximum similarity for each person

                    # Check if similarities list is not empty
                    if similarities:
                        best_match_index = np.argmax(similarities)

                        if similarities[best_match_index] > 0.7:
                            print(f"Face recognized as {self.known_face_names[best_match_index]} with {similarities[best_match_index] * 100:.2f}% similarity! Unlocking door...")
                            if self.arduino is not None:
                                self.arduino.unlock()
                            break  # Exit after unlocking the door
                        else:
                            print(f"Unknown face with {similarities[best_match_index] * 100:.2f}% similarity. Access denied.")
                except Exception as e:
                    print(f"Error during face comparison: {e}")

                # Stop after the specified capture time
                if time.time() - start_time >= capture_time:
                    print("Capture time exceeded. Stopping capture.")
                    break

            video_capture.release()
            cv2.destroyAllWindows()
            self.running = False

        # Start the capture in a separate thread
        self.running = True
        self.capture_thread = threading.Thread(target=capture_video)
        self.capture_thread.start()