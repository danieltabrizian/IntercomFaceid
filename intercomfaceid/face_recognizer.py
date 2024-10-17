import numpy as np
import insightface
import json
import os
import time
from datetime import datetime
import logging


class FaceRecognizer:
    def __init__(self, stream_manager):
        self.FACE_DATA_FILE = os.path.join("/config", "faces_data.json")
        self.known_face_encodings = []
        self.known_face_names = []
        self.model = insightface.app.FaceAnalysis(name="buffalo_sc")
        self.model.prepare(ctx_id=0)
        self.load_face_data()
        self.stream_manager = stream_manager
        self.arduino = None
        self.mqtt_client = None

    def set_arduino(self, arduino):
        self.arduino = arduino

    def set_mqtt_client(self, mqtt_client):
        self.mqtt_client = mqtt_client

    def cosine_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def save_face_data(self):
        face_data = {
            "encodings": [[face.tolist() for face in faces] for faces in self.known_face_encodings],
            "encodings": [[face.tolist() for face in faces] for faces in self.known_face_encodings],
            "names": self.known_face_names
        }
        with open(self.FACE_DATA_FILE, 'w') as f:
            json.dump(face_data, f)
        logging.info(f"Saved face data to {self.FACE_DATA_FILE}")

    def load_face_data(self):
        if os.path.exists(self.FACE_DATA_FILE):
            with open(self.FACE_DATA_FILE, 'r') as f:
                face_data = json.load(f)
                self.known_face_encodings = [[np.array(face) for face in faces] for faces in face_data["encodings"]]
                self.known_face_names = face_data["names"]
            logging.info(f"Loaded face data from {self.FACE_DATA_FILE}")
        else:
            logging.info("No face data file found, starting with an empty face database.")

    def get_face_embedding(self, image):
        small_image = cv2.resize(image, (320, 240))
        faces = self.model.get(small_image)
        if len(faces) > 0:
            embedding = faces[0].embedding
            return np.array(embedding)
        else:
            return None

    def captureFace(self, capture_time=30):
        if not self.stream_manager.is_capturing:
            logging.info("Stream is not running. Starting video stream...")
            if not self.stream_manager.start_video_stream():
                logging.error("Failed to start video stream. Cannot proceed with face capture.")
                return

        start_time = time.time()
        frame_processed = False
        frame_counter = 0
        frame_rate_start_time = time.time()  # Initialize the timer for frame rate calculation

        while time.time() - start_time < capture_time:
            ret, frame = self.stream_manager.get_frame()
            if not ret:
                continue
            
            frame_processed = True
            frame_counter += 1  # Increment the frame counter

            try:
                embedding_start_time = time.time()
                embedding = self.get_face_embedding(frame)

                 # Log frame rate every second
                current_time = time.time()
                if current_time - frame_rate_start_time >= 1.0:
                    fps = frame_counter / (current_time - frame_rate_start_time)  # Calculate frames per second
                    logging.info(f"Current frame rate during recognition: {fps:.2f} FPS")
                    frame_rate_start_time = current_time  # Reset the timer
                    frame_counter = 0  # Reset the frame counter

                if embedding is None:
                    continue

                similarities = []
                for i, known_faces in enumerate(self.known_face_encodings):
                    face_similarities = [self.cosine_similarity(embedding, known_face) for known_face in known_faces]
                    similarities.append(np.max(face_similarities))

                if similarities:
                    best_match_index = np.argmax(similarities)
                    if similarities[best_match_index] > 0.5:
                        logging.info(f"Face recognized as {self.known_face_names[best_match_index]} with {similarities[best_match_index] * 100:.2f}% similarity! Unlocking door...")
                        if self.arduino is not None:
                            self.arduino.unlock()
                        return  # Exit the method after successful recognition
                    else:
                        logging.info(f"Unknown face with {similarities[best_match_index] * 100:.2f}% similarity. Access denied.")



            except Exception as e:
                logging.error(f"Error during face comparison: {e}")

        if not frame_processed:
            logging.warning("No frames were processed during the capture period.")
        logging.warning("Face not recognized. Access denied.")

    
    def learn_new_face(self, person_name=None):
        if person_name is None:
            person_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        logging.info(f"Learning new face for {person_name}...")
        
        if not self.stream_manager.is_capturing:
            if not self.stream_manager.start_video_stream():
                logging.error("Failed to start video stream. Cannot learn new face.")
                return

        start_time = time.time()
        session_embeddings = []
        learning_duration = 5

        while time.time() - start_time < learning_duration:
            ret, frame = self.stream_manager.get_frame()
            if not ret:
                time.sleep(0.1)
                continue

            try:
                embedding = self.get_face_embedding(frame)

                if embedding is None:
                    logging.info("No face detected. Skipping frame...")
                    continue

                is_known_face = False
                for i, known_faces in enumerate(self.known_face_encodings):
                    for known_face in known_faces:
                        similarity = self.cosine_similarity(embedding, known_face)
                        if similarity > 0.7:
                            logging.info(f"Embedding matches known person {self.known_face_names[i]} with {similarity * 100:.2f}% similarity. Skipping this embedding.")
                            is_known_face = True
                            break
                    if is_known_face:
                        break

                if not is_known_face:
                    is_similar_to_session_embedding = False
                    for session_embedding in session_embeddings:
                        session_similarity = self.cosine_similarity(embedding, session_embedding)
                        if session_similarity > 0.7:
                            logging.info(f"New embedding is too similar to another embedding in this session ({session_similarity * 100:.2f}% similarity). Skipping this frame.")
                            is_similar_to_session_embedding = True
                            break

                    if not is_similar_to_session_embedding:
                        session_embeddings.append(embedding)
                        logging.info(f"Collected new embedding for {person_name}.")
                    else:
                        logging.info("Skipping this frame due to similarity with other session embeddings.")

            except Exception as e:
                logging.error(f"Error generating embedding: {e}")

        if session_embeddings:
            self.known_face_encodings.append(session_embeddings)
            self.known_face_names.append(person_name)
            logging.info(f"New face {person_name} learned and saved with {len(session_embeddings)} embeddings!")
        else:
            logging.info(f"No unique embeddings collected for {person_name}, face might already exist.")

        self.save_face_data()
        if self.arduino is not None:
            self.arduino.unlock()
        logging.info(f"Finished learning new face for {person_name}!")