import numpy as np
import insightface
import json
import os
import time
import threading
from datetime import datetime
import logging
import cv2

SFACE_DETECTOR_PATH = '/models/yunet.onnx'
SFACE_RECOGNIZER_PATH = '/models/sface.onnx'

FAST_PHASE_SECONDS = 10
FAST_THRESHOLD = 0.38    # SFace cosine similarity (0–1)
HEAVY_THRESHOLD = 0.50   # buffalo_sc cosine similarity (0–1)


class FaceRecognizer:
    def __init__(self, stream_manager, event_logger=None):
        self.FACE_DATA_FILE = '/config/faces_data.json'
        self.known_face_encodings = []
        self.known_face_names = []
        self.model_types = []   # 'sface' or 'buffalo_sc' per person
        self._lock = threading.Lock()
        self.stream_manager = stream_manager
        self.arduino = None
        self.mqtt_client = None
        self.event_logger = event_logger

        logging.info('Loading buffalo_sc (heavy model)...')
        self._heavy_model = insightface.app.FaceAnalysis(name='buffalo_sc')
        self._heavy_model.prepare(ctx_id=0)

        self._sface_ready = self._init_sface()
        self.load_face_data()

    # ------------------------------------------------------------------ models

    def _init_sface(self):
        if not os.path.exists(SFACE_DETECTOR_PATH) or not os.path.exists(SFACE_RECOGNIZER_PATH):
            logging.warning('SFace model files not found — fast phase disabled.')
            return False
        try:
            self._yunet = cv2.FaceDetectorYN.create(
                SFACE_DETECTOR_PATH, '', (320, 320),
                score_threshold=0.6, nms_threshold=0.3
            )
            self._sface = cv2.FaceRecognizerSF.create(SFACE_RECOGNIZER_PATH, '')
            logging.info('SFace fast model ready.')
            return True
        except Exception as e:
            logging.warning(f'Failed to load SFace models: {e} — fast phase disabled.')
            return False

    def _get_sface_embedding(self, frame):
        if not self._sface_ready:
            return None
        try:
            h, w = frame.shape[:2]
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(frame)
            if faces is None or len(faces) == 0:
                return None
            aligned = self._sface.alignCrop(frame, faces[0])
            return self._sface.feature(aligned)
        except Exception:
            return None

    def _sface_sim(self, e1, e2):
        return float(self._sface.match(e1, e2, cv2.FaceRecognizerSF.FR_COSINE))

    def _get_heavy_embedding(self, frame):
        small = cv2.resize(frame, (320, 240))
        faces = self._heavy_model.get(small)
        if not faces:
            return None
        return np.array(faces[0].embedding)

    def _heavy_sim(self, e1, e2):
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

    # used externally by old callers (learn_new_face dedup check)
    def cosine_similarity(self, e1, e2):
        return self._heavy_sim(e1, e2)

    # ---------------------------------------------------------------- storage

    def save_face_data(self):
        with self._lock:
            data = {
                'names': list(self.known_face_names),
                'encodings': [[e.tolist() for e in embs] for embs in self.known_face_encodings],
                'model_types': list(self.model_types),
            }
        with open(self.FACE_DATA_FILE, 'w') as f:
            json.dump(data, f)
        logging.info(f'Saved {len(data["names"])} faces to {self.FACE_DATA_FILE}')

    def load_face_data(self):
        if not os.path.exists(self.FACE_DATA_FILE):
            logging.info('No face data file, starting empty.')
            self.save_face_data()
            return
        with open(self.FACE_DATA_FILE, 'r') as f:
            data = json.load(f)
        with self._lock:
            self.known_face_names = data.get('names', [])
            self.known_face_encodings = [
                [np.array(e) for e in embs] for embs in data.get('encodings', [])
            ]
            # Existing entries without model_types default to buffalo_sc
            self.model_types = data.get(
                'model_types', ['buffalo_sc'] * len(self.known_face_names)
            )
        sface_count = self.model_types.count('sface')
        heavy_count = self.model_types.count('buffalo_sc')
        logging.info(f'Loaded {len(self.known_face_names)} faces: {sface_count} sface, {heavy_count} buffalo_sc')

    # ------------------------------------------------------------ wiring

    def set_arduino(self, arduino):
        self.arduino = arduino

    def set_mqtt_client(self, client):
        self.mqtt_client = client

    def _unlock_and_publish(self, name):
        if self.arduino:
            self.arduino.unlock()
        if self.mqtt_client:
            self.mqtt_client.publish_face_recognized(name)

    # --------------------------------------------------------- recognition

    def captureFace(self, capture_time=30, run_recognition=True):
        if not self.stream_manager.is_capturing:
            logging.info('Starting video stream...')
            if not self.stream_manager.start_video_stream():
                logging.error('Failed to start video stream.')
                return

        ret, frame = self.stream_manager.get_frame()
        if not ret:
            logging.warning('Could not grab frame for snapshot.')
            return

        snapshot_filename = None
        if self.event_logger is not None:
            snapshot_filename = self.event_logger.save_snapshot(frame, prefix='bell')
            self.event_logger.log('bell_ring', snapshot=snapshot_filename)

        if not run_recognition:
            return

        if self.event_logger is not None:
            self.event_logger.log('recognition_started')

        start_time = time.time()
        frame_buffer = [frame]
        fps_counter = 0
        fps_timer = time.time()
        embed_frames = 0
        embed_ms_total = 0.0
        match = None

        while time.time() - start_time < capture_time:
            ret, frame = self.stream_manager.get_frame()
            if not ret:
                continue

            frame_buffer.append(frame)
            if len(frame_buffer) > 80:
                frame_buffer.pop(0)

            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                logging.info(f'Recognition FPS: {fps_counter / (now - fps_timer):.1f}')
                fps_counter = 0
                fps_timer = now

            elapsed = now - start_time
            try:
                t0 = time.time()
                if elapsed < FAST_PHASE_SECONDS and self._sface_ready:
                    match = self._try_fast(frame)
                elif elapsed >= FAST_PHASE_SECONDS:
                    match = self._try_heavy(frame, list(frame_buffer))
                else:
                    continue
                embed_ms_total += (time.time() - t0) * 1000
                embed_frames += 1
            except Exception as e:
                logging.error(f'Recognition error: {e}')
                continue

            if match:
                break

        avg_ms = round(embed_ms_total / embed_frames, 1) if embed_frames else 0
        duration_s = round(time.time() - start_time, 1)

        if match:
            logging.info(f'[{match["model"]}] {match["name"]} {match["similarity"]*100:.1f}% — '
                         f'{avg_ms}ms/frame avg over {embed_frames} frames')
            if self.event_logger is not None:
                self.event_logger.log('face_recognized',
                                      name=match['name'],
                                      similarity=round(match['similarity'], 4),
                                      model=match['model'],
                                      snapshot=snapshot_filename,
                                      avg_ms_per_frame=avg_ms,
                                      frames=embed_frames,
                                      duration_s=duration_s)
            self._unlock_and_publish(match['name'])
            if match.get('migrate'):
                threading.Thread(
                    target=self._migrate_to_sface,
                    args=(match['name'], list(frame_buffer)),
                    daemon=True
                ).start()
        else:
            logging.warning(f'No face matched — {avg_ms}ms/frame avg over {embed_frames} frames')
            if self.event_logger is not None:
                self.event_logger.log('face_denied',
                                      similarity=None,
                                      snapshot=snapshot_filename,
                                      avg_ms_per_frame=avg_ms,
                                      frames=embed_frames,
                                      duration_s=duration_s)

    def _try_fast(self, frame):
        """Returns match dict or None. No side effects."""
        with self._lock:
            fast_idx = [i for i, m in enumerate(self.model_types) if m == 'sface']
            if not fast_idx:
                return None
            names = [self.known_face_names[i] for i in fast_idx]
            encodings = [self.known_face_encodings[i] for i in fast_idx]

        embedding = self._get_sface_embedding(frame)
        if embedding is None:
            return None

        best_score, best_name = 0.0, None
        for name, embs in zip(names, encodings):
            score = max(self._sface_sim(embedding, e) for e in embs)
            if score > best_score:
                best_score, best_name = score, name

        if best_score >= FAST_THRESHOLD:
            return {'name': best_name, 'similarity': best_score, 'model': 'sface', 'migrate': False}
        return None

    def _try_heavy(self, frame, frames_buffer):
        """Returns match dict (with migrate=True) or None. No side effects."""
        with self._lock:
            heavy_idx = [i for i, m in enumerate(self.model_types) if m == 'buffalo_sc']
            if not heavy_idx:
                return None
            names = [self.known_face_names[i] for i in heavy_idx]
            encodings = [self.known_face_encodings[i] for i in heavy_idx]

        embedding = self._get_heavy_embedding(frame)
        if embedding is None:
            return None

        similarities = [
            max(self._heavy_sim(embedding, e) for e in embs)
            for embs in encodings
        ]
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= HEAVY_THRESHOLD:
            return {'name': names[best_idx], 'similarity': best_score,
                    'model': 'buffalo_sc', 'migrate': True}
        return None

    # ---------------------------------------------------------- migration

    def _migrate_to_sface(self, name, frames):
        if not self._sface_ready:
            logging.warning(f'SFace not ready, skipping migration for {name}')
            return

        logging.info(f'Migrating {name} → SFace using {len(frames)} buffered frames...')
        new_embeddings = []
        for frame in frames:
            emb = self._get_sface_embedding(frame)
            if emb is None:
                continue
            # 0.50 threshold: keep embedding if it differs meaningfully from all stored ones.
            # SFace same-face similarity is typically 0.80-0.95, so 0.50 keeps diverse samples
            # without letting in faces from different people (recognition threshold is 0.38).
            if any(self._sface_sim(emb, e) > 0.50 for e in new_embeddings):
                continue
            new_embeddings.append(emb)
        logging.info(f'Migration extracted {len(new_embeddings)} unique SFace embeddings for {name}')

        if not new_embeddings:
            logging.warning(f'Migration failed for {name}: SFace found no face in buffered frames')
            return

        with self._lock:
            if name not in self.known_face_names:
                return
            idx = self.known_face_names.index(name)
            self.known_face_encodings[idx] = new_embeddings
            self.model_types[idx] = 'sface'

        self.save_face_data()
        if self.event_logger is not None:
            self.event_logger.log('face_migrated', name=name, embeddings=len(new_embeddings))
        logging.info(f'Migrated {name} → SFace ({len(new_embeddings)} embeddings)')

    # ------------------------------------------------------- enrollment

    def learn_new_face(self, person_name=None):
        if person_name is None:
            person_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        use_sface = self._sface_ready
        model_label = 'sface' if use_sface else 'buffalo_sc'
        logging.info(f'Learning {person_name} with {model_label}...')

        if not self.stream_manager.is_capturing:
            if not self.stream_manager.start_video_stream():
                logging.error('Failed to start video stream.')
                return

        start_time = time.time()
        session_embeddings = []
        face_snapshot_saved = False

        while time.time() - start_time < 5:
            ret, frame = self.stream_manager.get_frame()
            if not ret:
                time.sleep(0.1)
                continue
            try:
                embedding = self._get_sface_embedding(frame) if use_sface else self._get_heavy_embedding(frame)
                if embedding is None:
                    continue

                if not face_snapshot_saved and self.event_logger is not None:
                    self.event_logger.save_face_snapshot(frame, person_name)
                    face_snapshot_saved = True

                with self._lock:
                    all_encodings = list(self.known_face_encodings)
                    all_types = list(self.model_types)
                    all_names = list(self.known_face_names)

                # For heavy model keep buffalo_sc threshold (0.7); for sface use 0.50
                dedup_thresh = 0.50 if use_sface else 0.7

                is_known = False
                for embs, mtype, mname in zip(all_encodings, all_types, all_names):
                    if mtype != model_label:
                        continue
                    for e in embs:
                        sim = self._sface_sim(embedding, e) if use_sface else self._heavy_sim(embedding, e)
                        if sim > dedup_thresh:
                            logging.info(f'Matches existing {mname}, skipping frame.')
                            is_known = True
                            break
                    if is_known:
                        break

                if not is_known:
                    too_similar = any(
                        (self._sface_sim(embedding, e) if use_sface else self._heavy_sim(embedding, e)) > dedup_thresh
                        for e in session_embeddings
                    )
                    if not too_similar:
                        session_embeddings.append(embedding)
                        logging.info(f'Embedding #{len(session_embeddings)} for {person_name}')
            except Exception as e:
                logging.error(f'Error in learn_new_face: {e}')

        with self._lock:
            if session_embeddings:
                self.known_face_encodings.append(session_embeddings)
                self.known_face_names.append(person_name)
                self.model_types.append(model_label)
                logging.info(f'Enrolled {person_name} ({model_label}, {len(session_embeddings)} embeddings)')
            else:
                logging.warning(f'No embeddings collected for {person_name}')

        self.save_face_data()
        if self.arduino:
            self.arduino.unlock()

    # ---------------------------------------------------------- dashboard API

    def get_faces_info(self):
        with self._lock:
            names = list(self.known_face_names)
            encodings = list(self.known_face_encodings)
            types = list(self.model_types)
        return [
            {
                'name': n,
                'embedding_count': len(e),
                'model': t,
                'has_snapshot': self.event_logger.face_snapshot_exists(n) if self.event_logger else False,
            }
            for n, e, t in zip(names, encodings, types)
        ]

    def delete_face(self, name):
        with self._lock:
            if name not in self.known_face_names:
                return False
            idx = self.known_face_names.index(name)
            self.known_face_names.pop(idx)
            self.known_face_encodings.pop(idx)
            self.model_types.pop(idx)
        self.save_face_data()
        if self.event_logger is not None:
            snap = os.path.join(self.event_logger.face_snapshots_dir, f'{name}.jpg')
            if os.path.exists(snap):
                os.remove(snap)
        logging.info(f'Deleted face: {name}')
        return True
