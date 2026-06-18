import numpy as np
import insightface
import json
import os
import time
import threading
from datetime import datetime
import logging
import cv2


class FaceRecognizer:
    def __init__(self, stream_manager, event_logger=None):
        self.FACE_DATA_FILE = os.path.join("/config", "faces_data.json")
        self.known_face_encodings = []
        self.known_face_names = []
        self._lock = threading.Lock()
        self.model = insightface.app.FaceAnalysis(name="buffalo_sc")
        self.model.prepare(ctx_id=0)
        self.load_face_data()
        self.stream_manager = stream_manager
        self.arduino = None
        self.mqtt_client = None
        self.event_logger = event_logger

    def set_arduino(self, arduino):
        self.arduino = arduino

    def set_mqtt_client(self, mqtt_client):
        self.mqtt_client = mqtt_client

    def cosine_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def save_face_data(self):
        face_data = {
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
            self.save_face_data()

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
        frame_rate_start_time = time.time()
        snapshot_filename = None

        while time.time() - start_time < capture_time:
            ret, frame = self.stream_manager.get_frame()
            if not ret:
                continue

            frame_processed = True
            frame_counter += 1

            if snapshot_filename is None and self.event_logger is not None:
                snapshot_filename = self.event_logger.save_snapshot(frame, prefix='bell')
                self.event_logger.log('bell_ring', snapshot=snapshot_filename)

            try:
                embedding = self.get_face_embedding(frame)

                current_time = time.time()
                if current_time - frame_rate_start_time >= 1.0:
                    fps = frame_counter / (current_time - frame_rate_start_time)
                    logging.info(f"Current frame rate during recognition: {fps:.2f} FPS")
                    frame_rate_start_time = current_time
                    frame_counter = 0

                if embedding is None:
                    continue

                with self._lock:
                    names = list(self.known_face_names)
                    encodings = list(self.known_face_encodings)

                similarities = []
                for known_faces in encodings:
                    face_similarities = [self.cosine_similarity(embedding, known_face) for known_face in known_faces]
                    similarities.append(np.max(face_similarities))

                if similarities:
                    best_match_index = np.argmax(similarities)
                    score = float(similarities[best_match_index])
                    if score > 0.5:
                        name = names[best_match_index]
                        logging.info(f"Face recognized as {name} with {score * 100:.2f}% similarity! Unlocking door...")
                        if self.event_logger is not None:
                            self.event_logger.log('face_recognized', name=name, similarity=round(score, 4), snapshot=snapshot_filename)
                        if self.arduino is not None:
                            self.arduino.unlock()
                        self.mqtt_client.publish_face_recognized(name)
                        return

                    logging.info(f"Unknown face with {score * 100:.2f}% similarity. Access denied.")
                    if self.event_logger is not None:
                        self.event_logger.log('face_denied', similarity=round(score, 4), snapshot=snapshot_filename)

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
        face_snapshot_saved = False

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

                if not face_snapshot_saved and self.event_logger is not None:
                    self.event_logger.save_face_snapshot(frame, person_name)
                    face_snapshot_saved = True

                with self._lock:
                    known_names = list(self.known_face_names)
                    known_encodings = list(self.known_face_encodings)

                is_known_face = False
                for i, known_faces in enumerate(known_encodings):
                    for known_face in known_faces:
                        similarity = self.cosine_similarity(embedding, known_face)
                        if similarity > 0.7:
                            logging.info(f"Embedding matches known person {known_names[i]} with {similarity * 100:.2f}% similarity. Skipping this embedding.")
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

        with self._lock:
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

    def get_faces_info(self):
        with self._lock:
            names = list(self.known_face_names)
            encodings = list(self.known_face_encodings)
        result = []
        for name, embs in zip(names, encodings):
            has_snapshot = self.event_logger.face_snapshot_exists(name) if self.event_logger else False
            result.append({'name': name, 'embedding_count': len(embs), 'has_snapshot': has_snapshot})
        return result

    def delete_face(self, name):
        with self._lock:
            if name not in self.known_face_names:
                return False
            idx = self.known_face_names.index(name)
            self.known_face_names.pop(idx)
            self.known_face_encodings.pop(idx)
        self.save_face_data()
        if self.event_logger is not None:
            snap_path = os.path.join(self.event_logger.face_snapshots_dir, f'{name}.jpg')
            if os.path.exists(snap_path):
                os.remove(snap_path)
        logging.info(f"Deleted face: {name}")
        return True