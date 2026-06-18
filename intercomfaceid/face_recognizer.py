import numpy as np
import insightface
from insightface.utils import face_align
import json
import os
import time
import threading
from datetime import datetime
import logging
import cv2

MATCH_THRESHOLD = 0.50   # buffalo_sc cosine similarity to accept a match (0–1)
DEDUP_THRESHOLD = 0.70   # during enrollment, skip embeddings more similar than this
BLUR_THRESHOLD  = 80.0   # Laplacian variance on the face crop; below this = "blurry".
                         # Gates the EXPENSIVE embedding step — detection still runs.
FORCE_AFTER_MS  = 100    # failsafe: if no frame has been embedded in this long, embed
                         # the next detected face regardless of blur, so recognition is
                         # never starved. At 7 FPS this means ~every detected face is
                         # embedded; raise it (and BLUR_THRESHOLD) once calibration data
                         # shows where real matches cluster.


class FaceRecognizer:
    """buffalo_sc (InsightFace) only. Pipeline per frame:
        detect (SCRFD, ~cheap) -> face-crop blur gate -> embed (ArcFace, expensive)
    Detection runs on every frame; the costly embedding only runs on sharp frames.
    SCRFD returns full-resolution landmarks, so the aligned recognition crop is taken
    from the full-res frame (best embedding quality) while detection stays cheap."""

    def __init__(self, stream_manager, event_logger=None, blur_calibration=None):
        self.FACE_DATA_FILE = '/config/faces_data.json'
        self.known_face_encodings = []
        self.known_face_names = []
        self._lock = threading.Lock()
        self.stream_manager = stream_manager
        self.arduino = None
        self.mqtt_client = None
        self.event_logger = event_logger
        self.blur_calibration = blur_calibration
        self._logged_res = False

        try:
            cores = os.cpu_count() or 2
            cv2.setNumThreads(cores)
            logging.info(f'OpenCV thread count set to {cores}')
        except Exception as e:
            logging.warning(f'Could not set OpenCV thread count: {e}')

        logging.info('Loading buffalo_sc...')
        self._model = insightface.app.FaceAnalysis(name='buffalo_sc')
        # det_size bounds detection cost regardless of input frame size.
        self._model.prepare(ctx_id=0, det_size=(320, 320))
        self._det = self._model.models['detection']
        self._rec = self._model.models['recognition']

        self.load_face_data()

    # ------------------------------------------------------------ detection / embedding

    def _detect(self, frame):
        """Run SCRFD. Returns (bbox[x1,y1,x2,y2,score], kps[5,2]) of the largest
        face, or None. Coordinates are in full-resolution frame space."""
        bboxes, kpss = self._det.detect(frame, max_num=0, metric='default')
        if bboxes is None or bboxes.shape[0] == 0:
            return None
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        i = int(np.argmax(areas))
        return bboxes[i], kpss[i]

    def _crop_sharpness(self, frame, bbox):
        """Laplacian variance of the face crop only (reliable on a static camera,
        where a sharp background would otherwise mask a blurry face)."""
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(x2, frame.shape[1])
        y2 = min(y2, frame.shape[0])
        if x2 - x1 < 10 or y2 - y1 < 10:
            return 999.0  # too small to judge — treat as sharp
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _embed(self, frame, kps):
        """Align the face from the full-res frame and run the ArcFace embedding."""
        aligned = face_align.norm_crop(frame, landmark=kps, image_size=112)
        feat = self._rec.get_feat(aligned)
        return feat[0] if getattr(feat, 'ndim', 1) == 2 else feat

    def _sim(self, e1, e2):
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

    # kept for any external callers
    def cosine_similarity(self, e1, e2):
        return self._sim(e1, e2)

    def _match(self, embedding):
        with self._lock:
            names = list(self.known_face_names)
            encodings = list(self.known_face_encodings)
        if not names:
            return None
        sims = [max(self._sim(embedding, e) for e in embs) for embs in encodings]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= MATCH_THRESHOLD:
            return {'name': names[best_idx], 'similarity': best_score}
        return None

    # ---------------------------------------------------------------- storage

    def save_face_data(self):
        with self._lock:
            data = {
                'names': list(self.known_face_names),
                'encodings': [[e.tolist() for e in embs] for embs in self.known_face_encodings],
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

        names = data.get('names', [])
        encodings = [[np.array(e) for e in embs] for embs in data.get('encodings', [])]
        # Legacy files may carry per-person model_types. Any 'sface' entries hold
        # embeddings in SFace's vector space, which is incompatible with buffalo_sc —
        # drop them (they'd need re-enrollment) so they can't corrupt matching.
        model_types = data.get('model_types')
        dropped = 0
        if model_types:
            keep_n, keep_e = [], []
            for n, e, t in zip(names, encodings, model_types):
                if t == 'sface':
                    dropped += 1
                    continue
                keep_n.append(n)
                keep_e.append(e)
            names, encodings = keep_n, keep_e

        with self._lock:
            self.known_face_names = names
            self.known_face_encodings = encodings
        logging.info(f'Loaded {len(names)} faces' +
                     (f' (dropped {dropped} incompatible SFace entries — re-enroll them)' if dropped else ''))
        if dropped:
            self.save_face_data()  # rewrite without model_types / sface entries

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

        if not self._logged_res:
            logging.info(f'Source frame resolution: {frame.shape[1]}x{frame.shape[0]}')
            self._logged_res = True

        with self._lock:
            have_faces = len(self.known_face_names) > 0
        if not have_faces:
            logging.warning('No faces enrolled — nothing to recognize.')
            if self.event_logger is not None:
                self.event_logger.log('face_denied', similarity=None, snapshot=snapshot_filename)
            return

        start_time = time.time()
        fps_counter = 0
        fps_timer = time.time()
        detect_ms, detect_frames = 0.0, 0
        embed_ms, embed_frames = 0.0, 0
        samples = []            # (sharpness, matched) for embedded frames
        forced_processed = 0    # embedded despite being below blur threshold (failsafe)
        auto_processed = 0      # embedded because at/above threshold
        skipped_blurry = 0      # below threshold AND failsafe didn't fire
        no_face_frames = 0      # frames where SCRFD found no face
        last_embedded = start_time
        match = None

        while time.time() - start_time < capture_time:
            ret, frame = self.stream_manager.get_frame()
            if not ret:
                continue

            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                logging.info(f'Recognition FPS: {fps_counter / (now - fps_timer):.1f}')
                fps_counter = 0
                fps_timer = now

            # --- Detection (cheap, every frame) ---
            try:
                t0 = time.time()
                det = self._detect(frame)
                detect_ms += (time.time() - t0) * 1000
                detect_frames += 1
            except Exception as e:
                logging.error(f'Detection error: {e}')
                continue
            if det is None:
                no_face_frames += 1
                continue
            bbox, kps = det
            sharpness = self._crop_sharpness(frame, bbox)

            # --- Blur gate (decides whether to pay for the embedding) ---
            forced = False
            if sharpness < BLUR_THRESHOLD:
                if (now - last_embedded) * 1000 >= FORCE_AFTER_MS:
                    forced = True   # failsafe: don't starve recognition
                else:
                    skipped_blurry += 1
                    continue

            # --- Embedding (expensive, sharp frames only) ---
            last_embedded = now
            try:
                t0 = time.time()
                emb = self._embed(frame, kps)
                embed_ms += (time.time() - t0) * 1000
                embed_frames += 1
                match = self._match(emb)
            except Exception as e:
                logging.error(f'Embedding error: {e}')
                continue

            samples.append((sharpness, bool(match)))
            if forced:
                forced_processed += 1
            else:
                auto_processed += 1

            if match:
                break

        duration_s = round(time.time() - start_time, 1)
        timing = {
            'detect_avg_ms': round(detect_ms / detect_frames, 1) if detect_frames else None,
            'detect_frames': detect_frames,
            'embed_avg_ms':  round(embed_ms / embed_frames, 1) if embed_frames else None,
            'embed_frames':  embed_frames,
            'forced_processed': forced_processed,
            'auto_processed': auto_processed,
            'skipped_blurry': skipped_blurry,
            'no_face_frames': no_face_frames,
            'duration_s':   duration_s,
        }

        if self.blur_calibration is not None:
            try:
                self.blur_calibration.record_batch(
                    samples, forced_processed, auto_processed, skipped_blurry,
                    blur_threshold=BLUR_THRESHOLD, force_after_ms=FORCE_AFTER_MS)
            except Exception as e:
                logging.error(f'Failed to record blur calibration: {e}')

        summary = (f'detect {timing["detect_avg_ms"]}ms×{detect_frames}f  '
                   f'embed {timing["embed_avg_ms"]}ms×{embed_frames}f  '
                   f'(forced {forced_processed}, auto {auto_processed}, '
                   f'skipped {skipped_blurry}, no_face {no_face_frames})')

        if match:
            logging.info(f'Recognized {match["name"]} {match["similarity"]*100:.1f}% — {summary}')
            if self.event_logger is not None:
                self.event_logger.log('face_recognized',
                                      name=match['name'],
                                      similarity=round(match['similarity'], 4),
                                      model='buffalo_sc',
                                      snapshot=snapshot_filename,
                                      **timing)
            self._unlock_and_publish(match['name'])
        else:
            logging.warning(f'No face matched — {summary}')
            if self.event_logger is not None:
                self.event_logger.log('face_denied',
                                      similarity=None,
                                      snapshot=snapshot_filename,
                                      **timing)

    # ---------------------------------------------------------- benchmark

    def benchmark(self, iterations=20):
        """Measure raw per-stage latency on live frames, as if a face were present.
        Detection runs on each frame; the embedding is forced on a synthetic crop
        when no real face is detected, so timing reflects the full pipeline."""
        if not self.stream_manager.is_capturing:
            self.stream_manager.start_video_stream()

        frames = []
        deadline = time.time() + 12
        while len(frames) < iterations and time.time() < deadline:
            ret, frame = self.stream_manager.get_frame()
            if ret:
                frames.append(frame)
            else:
                time.sleep(0.05)
        if not frames:
            return {'error': 'No frames available from stream'}

        h, w = frames[0].shape[:2]

        def avg(lst):
            return round(sum(lst) / len(lst), 1) if lst else None

        det_times, emb_times = [], []
        faces_detected = 0

        for f in frames:
            fh, fw = f.shape[:2]
            t = time.time()
            det = self._detect(f)
            det_times.append((time.time() - t) * 1000)

            if det is not None:
                faces_detected += 1
                _, kps = det
                t = time.time()
                self._embed(f, kps)
                emb_times.append((time.time() - t) * 1000)
            else:
                # force embedding on a centered crop so we still time it
                s = min(fh, fw)
                cy, cx = fh // 2, fw // 2
                crop = cv2.resize(f[cy - s // 2:cy + s // 2, cx - s // 2:cx + s // 2], (112, 112))
                t = time.time()
                self._rec.get_feat(crop)
                emb_times.append((time.time() - t) * 1000)

        d, e = avg(det_times), avg(emb_times)
        result = {
            'frames': len(frames),
            'resolution': f'{w}x{h}',
            'faces_detected': faces_detected,
            'detect_ms': d,
            'embed_ms': e,
            'total_ms': round((d or 0) + (e or 0), 1),
        }
        logging.info(f'Benchmark: {result}')
        return result

    # ------------------------------------------------------- enrollment

    def learn_new_face(self, person_name=None):
        if person_name is None:
            person_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        logging.info(f'Learning {person_name}...')

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
                det = self._detect(frame)
                if det is None:
                    continue
                bbox, kps = det
                if self._crop_sharpness(frame, bbox) < BLUR_THRESHOLD:
                    continue  # don't enroll blurry frames
                embedding = self._embed(frame, kps)

                if not face_snapshot_saved and self.event_logger is not None:
                    self.event_logger.save_face_snapshot(frame, person_name)
                    face_snapshot_saved = True

                with self._lock:
                    known = list(self.known_face_encodings)
                    known_names = list(self.known_face_names)

                # Skip if this already matches an enrolled person
                is_known = False
                for embs, mname in zip(known, known_names):
                    if any(self._sim(embedding, e) > DEDUP_THRESHOLD for e in embs):
                        logging.info(f'Matches existing {mname}, skipping frame.')
                        is_known = True
                        break
                if is_known:
                    continue

                # Skip near-duplicates within this session
                if any(self._sim(embedding, e) > DEDUP_THRESHOLD for e in session_embeddings):
                    continue
                session_embeddings.append(embedding)
                logging.info(f'Embedding #{len(session_embeddings)} for {person_name}')
            except Exception as e:
                logging.error(f'Error in learn_new_face: {e}')

        with self._lock:
            if session_embeddings:
                self.known_face_encodings.append(session_embeddings)
                self.known_face_names.append(person_name)
                logging.info(f'Enrolled {person_name} ({len(session_embeddings)} embeddings)')
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
        return [
            {
                'name': n,
                'embedding_count': len(e),
                'has_snapshot': self.event_logger.face_snapshot_exists(n) if self.event_logger else False,
            }
            for n, e in zip(names, encodings)
        ]

    def delete_face(self, name):
        with self._lock:
            if name not in self.known_face_names:
                return False
            idx = self.known_face_names.index(name)
            self.known_face_names.pop(idx)
            self.known_face_encodings.pop(idx)
        self.save_face_data()
        if self.event_logger is not None:
            snap = os.path.join(self.event_logger.face_snapshots_dir, f'{name}.jpg')
            if os.path.exists(snap):
                os.remove(snap)
        logging.info(f'Deleted face: {name}')
        return True
