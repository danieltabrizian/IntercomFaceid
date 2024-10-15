import cv2
import numpy as np
import insightface
import json
import os
import time
import threading
from datetime import datetime
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import io



class FaceRecognizer:
    class MJPEGStreamHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    while True:
                        jpg = self.server.face_recognizer.get_jpg_frame()
                        if jpg is None:
                            time.sleep(0.1)
                            continue
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-length', len(jpg))
                        self.end_headers()
                        self.wfile.write(jpg)
                        self.wfile.write(b'\r\n')
                except Exception as e:
                    logging.error(f"Streaming client {self.client_address} disconnected: {str(e)}")
            elif self.path == '/health':
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b"Stream is running")
            else:
                self.send_error(404)
                self.end_headers()


        def log_message(self, format, *args):
            # Suppress default logging to reduce noise
            return

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        def __init__(self, face_recognizer, *args, **kwargs):
            self.face_recognizer = face_recognizer
            super().__init__(*args, **kwargs)
    
    def __init__(self, stream_port=8080):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        self.FACE_DATA_FILE = "/config/faces_data.json"
        self.known_face_encodings = []
        self.known_face_names = []
        self.model = insightface.app.FaceAnalysis()
        self.model.prepare(ctx_id=0)
        self.load_face_data()

        self.video_capture = None
        self.is_capturing = False
        self.lock = threading.Lock()
        self.stream_thread = None
        self.face_detection_active = False
        self.stream_port = stream_port
        self.current_frame = None

        self.arduino = None
        self.mqtt_client = None

        # Start the video stream immediately
        self.start_video_stream()

        # Start the MJPEG server
        self.start_mjpeg_server()

    def set_arduino(self, arduino):
        self.arduino = arduino

    def set_mqtt_client(self, mqtt_client):
        self.mqtt_client = mqtt_client

    def start_video_stream(self):
        if self.stream_thread is None or not self.stream_thread.is_alive():
            self.stream_thread = threading.Thread(target=self._stream_video)
            self.stream_thread.daemon = True
            self.stream_thread.start()

    def _stream_video(self):
        while True:
            try:
                if self.video_capture is None or not self.video_capture.isOpened():
                    self.video_capture = cv2.VideoCapture(0)
                    if not self.video_capture.isOpened():
                        logging.error("Failed to open the camera. Retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    logging.info("Video stream started.")

                while True:
                    ret, frame = self.video_capture.read()
                    if not ret:
                        logging.error("Failed to capture frame. Restarting stream...")
                        break

                    with self.lock:
                        self.current_frame = frame

                    if self.face_detection_active:
                        self._process_frame(frame)

            except Exception as e:
                logging.error(f"Error in video stream: {e}. Restarting stream...")
                if self.video_capture is not None:
                    self.video_capture.release()
                    self.video_capture = None
                time.sleep(5)

    def get_jpg_frame(self):
        with self.lock:
            if self.current_frame is not None:
                ret, jpg = cv2.imencode('.jpg', self.current_frame)
                if ret:
                    return jpg.tobytes()
        return None

    def start_mjpeg_server(self):
        try:
            server = self.ThreadedHTTPServer(self, ('0.0.0.0', self.stream_port), self.MJPEGStreamHandler)
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            logging.info(f"MJPEG server started on port {self.stream_port}")
            
            # Get the actual IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # doesn't even have to be reachable
                s.connect(('10.255.255.255', 1))
                ip = s.getsockname()[0]
            except Exception:
                ip = '127.0.0.1'
            finally:
                s.close()
            
            logging.info(f"Stream available at http://{ip}:{self.stream_port}")
            logging.info(f"Health check available at http://{ip}:{self.stream_port}/health")
        except Exception as e:
            logging.error(f"Failed to start MJPEG server: {e}")


    def _process_frame(self, frame):
        try:
            embedding = self.get_face_embedding(frame)
            if embedding is None:
                return

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
                    self.face_detection_active = False
                else:
                    logging.info(f"Unknown face with {similarities[best_match_index] * 100:.2f}% similarity. Access denied.")

        except Exception as e:
            logging.error(f"Error during face comparison: {e}")

    def activate_face_detection(self, duration=30):
        self.face_detection_active = True
        threading.Timer(duration, self._deactivate_face_detection).start()

    def _deactivate_face_detection(self):
        self.face_detection_active = False

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

    def get_face_embedding(self, image):
        faces = self.model.get(image)
        if len(faces) > 0:
            embedding = faces[0].embedding
            return np.array(embedding)
        else:
            return None

    def learn_new_face(self, person_name=None):
        if person_name is None:
            person_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        logging.info(f"Learning new face for {person_name}...")
        
        start_time = time.time()
        session_embeddings = []

        while time.time() - start_time < 10:
            with self.lock:
                frame = self.current_frame.copy() if self.current_frame is not None else None
            
            if frame is None:
                logging.error("Failed to get current frame")
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

    def __del__(self):
        if self.video_capture is not None:
            self.video_capture.release()