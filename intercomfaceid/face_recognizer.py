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
REQUIRED_MATCHES = 2     # consecutive live frames that must match the SAME person
                         # before unlocking — guards against single-frame false hits
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

    def _face_crop_img(self, frame, bbox, pad=0.35):
        """A padded square-ish face crop for the gallery thumbnails."""
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad), int(bh * pad)
        x1 = max(0, x1 - px); y1 = max(0, y1 - py)
        x2 = min(frame.shape[1], x2 + px); y2 = min(frame.shape[0], y2 + py)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        return frame[y1:y2, x1:x2].copy()

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
        """Start the stream on demand, capture, then stop it again so the add-on
        consumes no CPU decoding frames while idle."""
        started_here = not self.stream_manager.is_capturing
        if started_here:
            logging.info('Starting video stream...')
            if not self.stream_manager.start_video_stream():
                logging.error('Failed to start video stream.')
                return
        try:
            self._do_capture(capture_time, run_recognition)
        finally:
            if started_here:
                self.stream_manager.stop_video_stream()

    def _do_capture(self, capture_time, run_recognition):
        # Cold start: wait briefly for the first decoded frame after (re)connecting.
        frame = None
        wait_until = time.time() + 4
        while time.time() < wait_until:
            ret, f = self.stream_manager.get_frame()
            if ret:
                frame = f
                break
            time.sleep(0.05)
        if frame is None:
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
        no_face_frames = 0      # frames where SCRFD found no face
        match = None            # set only once a match is CONFIRMED (see below)
        pending_name = None     # person matched on the previous frame
        streak = 0              # consecutive frames matching pending_name

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

            # --- Detection (every frame) ---
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
                pending_name = None   # a no-face frame breaks the streak
                streak = 0
                continue
            bbox, kps = det

            # --- Embedding + match (no blur gate — simple and reliable) ---
            try:
                t0 = time.time()
                emb = self._embed(frame, kps)
                embed_ms += (time.time() - t0) * 1000
                embed_frames += 1
                m = self._match(emb)
            except Exception as e:
                logging.error(f'Embedding error: {e}')
                continue

            # Require REQUIRED_MATCHES consecutive frames of the SAME person before
            # accepting — a single-frame fluke can't unlock the door.
            if m:
                if m['name'] == pending_name:
                    streak += 1
                else:
                    pending_name = m['name']
                    streak = 1
                if streak >= REQUIRED_MATCHES:
                    match = m
                    # Auto-refresh the person's gallery with this fresh face crop.
                    if self.event_logger is not None:
                        try:
                            crop = self._face_crop_img(frame, bbox)
                            if crop is not None:
                                self.event_logger.add_face_image(crop, match['name'])
                        except Exception as e:
                            logging.debug(f'gallery update failed: {e}')
                    break
            else:
                pending_name = None   # a non-matching face breaks the streak
                streak = 0

        duration_s = round(time.time() - start_time, 1)
        timing = {
            'detect_avg_ms': round(detect_ms / detect_frames, 1) if detect_frames else None,
            'detect_frames': detect_frames,
            'embed_avg_ms':  round(embed_ms / embed_frames, 1) if embed_frames else None,
            'embed_frames':  embed_frames,
            'no_face_frames': no_face_frames,
            'duration_s':   duration_s,
        }

        summary = (f'detect {timing["detect_avg_ms"]}ms×{detect_frames}f  '
                   f'embed {timing["embed_avg_ms"]}ms×{embed_frames}f  (no_face {no_face_frames})')

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
        started_here = not self.stream_manager.is_capturing
        if started_here:
            if not self.stream_manager.start_video_stream():
                logging.error('Failed to start video stream.')
                return
        try:
            self._do_learn(person_name)
        finally:
            if started_here:
                self.stream_manager.stop_video_stream()

    def _do_learn(self, person_name=None):
        if person_name is None:
            person_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        logging.info(f'Learning {person_name}...')

        start_time = time.time()
        session_embeddings = []

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

                # Seed the gallery with a crop from each accepted (diverse) frame
                if self.event_logger is not None:
                    crop = self._face_crop_img(frame, bbox)
                    if crop is not None:
                        self.event_logger.add_face_image(crop, person_name)
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
        result = []
        for n, e in zip(names, encodings):
            images = self.event_logger.face_images(n) if self.event_logger else []
            result.append({
                'name': n,
                'embedding_count': len(e),
                'images': images,
                'has_snapshot': len(images) > 0,
            })
        return result

    def rename_face(self, old, new):
        new = (new or '').strip()
        if not new:
            return {'success': False, 'error': 'Name cannot be empty'}
        with self._lock:
            if old not in self.known_face_names:
                return {'success': False, 'error': 'Person not found'}
            if new != old and new in self.known_face_names:
                return {'success': False, 'error': 'A person with that name already exists'}
            idx = self.known_face_names.index(old)
            self.known_face_names[idx] = new
        if self.event_logger is not None:
            try:
                self.event_logger.rename_face_images(old, new)
            except Exception as e:
                logging.error(f'Failed to rename face images: {e}')
        self.save_face_data()
        logging.info(f'Renamed face: {old} -> {new}')
        return {'success': True}

    def delete_face(self, name):
        with self._lock:
            if name not in self.known_face_names:
                return False
            idx = self.known_face_names.index(name)
            self.known_face_names.pop(idx)
            self.known_face_encodings.pop(idx)
        self.save_face_data()
        if self.event_logger is not None:
            self.event_logger.delete_face_images(name)
        logging.info(f'Deleted face: {name}')
        return True
