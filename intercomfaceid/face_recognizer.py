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
FAST_THRESHOLD  = 0.38   # SFace cosine similarity (0–1)
HEAVY_THRESHOLD = 0.50   # buffalo_sc cosine similarity (0–1)
BLUR_THRESHOLD  = 80.0   # Laplacian variance on the face crop; below this = "blurry"
FORCE_AFTER_MS  = 100    # if no frame processed in this long, process the next one
                         # regardless of blur (failsafe — never starve recognition).
                         # At 7 FPS (~143ms/frame) this means ~every detected face is
                         # processed, which is intentional during calibration.
DETECT_MAX_SIDE = 0      # 0 = disabled (detect on full-res frame). Set to e.g. 320 to
                         # run YuNet on a downscaled copy for speed (crop still aligned
                         # from full-res). Left OFF for now to measure raw per-model
                         # latency; use the /api/benchmark button to compare full vs 320.


class FaceRecognizer:
    def __init__(self, stream_manager, event_logger=None, blur_calibration=None):
        self.FACE_DATA_FILE = '/config/faces_data.json'
        self.known_face_encodings = []
        self.known_face_names = []
        self.model_types = []   # 'sface' or 'buffalo_sc' per person
        self._lock = threading.Lock()
        self.stream_manager = stream_manager
        self.arduino = None
        self.mqtt_client = None
        self.event_logger = event_logger
        self.blur_calibration = blur_calibration
        self._logged_res = False

        # Use all available CPU cores for OpenCV DNN (YuNet/SFace) ops.
        try:
            cores = os.cpu_count() or 4
            cv2.setNumThreads(cores)
            logging.info(f'OpenCV thread count set to {cores}')
        except Exception as e:
            logging.warning(f'Could not set OpenCV thread count: {e}')

        logging.info('Loading buffalo_sc (heavy model)...')
        self._heavy_model = insightface.app.FaceAnalysis(name='buffalo_sc')
        # det_size defaults to 640x640 in insightface; we feed small frames so a
        # 320x320 detector input is plenty and roughly halves detection cost.
        self._heavy_model.prepare(ctx_id=0, det_size=(320, 320))

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
        """Detect + blur-gate + embed in one call. Used by enrollment."""
        if not self._sface_ready:
            return None
        try:
            faces, sharpness = self._detect_face(frame)
            if faces is None:
                return None
            if sharpness < BLUR_THRESHOLD:
                logging.debug('SFace: blurry face crop, skipping embedding')
                return None
            return self._sface_embed(frame, faces)
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

    def _face_crop_sharpness(self, frame, face_bbox):
        """Laplacian variance on the detected face crop. ~0.3ms.
        Much more reliable than whole-frame on a static camera — the static
        background dominates whole-frame variance and would mask a blurry face."""
        x, y, bw, bh = [int(v) for v in face_bbox[:4]]
        x, y = max(0, x), max(0, y)
        bw = min(bw, frame.shape[1] - x)
        bh = min(bh, frame.shape[0] - y)
        if bw < 10 or bh < 10:
            return 999.0  # crop too small to judge — treat as sharp
        gray = cv2.cvtColor(frame[y:y+bh, x:x+bw], cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _detect_face(self, frame):
        """YuNet detection on a downscaled copy (fast), with coordinates scaled
        back to full resolution. SFace then aligns/crops from the sharp full-res
        frame, so detection gets cheaper without hurting embedding quality.
        Returns (faces, face_crop_sharpness) or (None, None)."""
        h, w = frame.shape[:2]
        scale = 1.0
        det_frame = frame
        longest = max(h, w)
        if DETECT_MAX_SIDE and longest > DETECT_MAX_SIDE:
            scale = DETECT_MAX_SIDE / longest
            det_frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
        dh, dw = det_frame.shape[:2]
        self._yunet.setInputSize((dw, dh))
        _, faces = self._yunet.detect(det_frame)
        if faces is None or len(faces) == 0:
            return None, None
        face = faces[0].astype('float32').copy()
        if scale != 1.0:
            face[:14] = face[:14] / scale   # bbox (4) + 5 landmarks (10) → full res
        return face.reshape(1, -1), self._face_crop_sharpness(frame, face)

    def _sface_embed(self, frame, faces):
        aligned = self._sface.alignCrop(frame, faces[0])
        return self._sface.feature(aligned)

    def _match_sface(self, embedding):
        with self._lock:
            idxs = [i for i, m in enumerate(self.model_types) if m == 'sface']
            if not idxs:
                return None
            names = [self.known_face_names[i] for i in idxs]
            encodings = [self.known_face_encodings[i] for i in idxs]
        best_score, best_name = 0.0, None
        for name, embs in zip(names, encodings):
            score = max(self._sface_sim(embedding, e) for e in embs)
            if score > best_score:
                best_score, best_name = score, name
        if best_score >= FAST_THRESHOLD:
            return {'name': best_name, 'similarity': best_score, 'model': 'sface', 'migrate': False}
        return None

    def _match_heavy(self, embedding):
        with self._lock:
            idxs = [i for i, m in enumerate(self.model_types) if m == 'buffalo_sc']
            if not idxs:
                return None
            names = [self.known_face_names[i] for i in idxs]
            encodings = [self.known_face_encodings[i] for i in idxs]
        sims = [max(self._heavy_sim(embedding, e) for e in embs) for embs in encodings]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= HEAVY_THRESHOLD:
            return {'name': names[best_idx], 'similarity': best_score,
                    'model': 'buffalo_sc', 'migrate': True}
        return None

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

        if not self._logged_res:
            logging.info(f'Source frame resolution: {frame.shape[1]}x{frame.shape[0]}')
            self._logged_res = True

        with self._lock:
            n_sface = self.model_types.count('sface')
            n_heavy = self.model_types.count('buffalo_sc')

        start_time = time.time()
        frame_buffer = [frame]
        fps_counter = 0
        fps_timer = time.time()
        fast_ms, fast_frames = 0.0, 0
        heavy_ms, heavy_frames = 0.0, 0
        # Blur-gate instrumentation
        samples = []            # (sharpness, matched) for processed frames
        forced_processed = 0    # below threshold but processed anyway (failsafe)
        auto_processed = 0      # at/above threshold
        skipped_blurry = 0      # below threshold AND failsafe didn't fire
        no_face_frames = 0      # frames where YuNet found no face
        last_processed = start_time
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
            # Fast phase only applies if there are migrated SFace faces to compare against.
            in_fast = self._sface_ready and n_sface > 0 and elapsed < FAST_PHASE_SECONDS
            # Heavy runs whenever fast isn't running and there are buffalo_sc faces —
            # crucially including from t=0 when no SFace faces exist yet, otherwise the
            # first FAST_PHASE_SECONDS would be wasted idling.
            in_heavy = (not in_fast) and n_heavy > 0

            if not in_fast and not in_heavy:
                # Nothing can match in the current phase. If nothing will ever match
                # (no heavy faces, and fast window is over or there are no sface faces),
                # stop early instead of spinning the rest of the window.
                if n_heavy == 0 and (n_sface == 0 or elapsed >= FAST_PHASE_SECONDS):
                    break
                continue

            # If SFace models aren't available we can't run the YuNet-based blur gate;
            # fall back to ungated heavy recognition.
            if not self._sface_ready:
                try:
                    t0 = time.time()
                    emb = self._get_heavy_embedding(frame)
                    match = self._match_heavy(emb) if emb is not None else None
                    heavy_ms += (time.time() - t0) * 1000
                    heavy_frames += 1
                except Exception as e:
                    logging.error(f'Recognition error: {e}')
                    continue
                if match:
                    break
                continue

            # --- Shared detection + blur gate ---
            try:
                t_detect = time.time()
                faces, sharpness = self._detect_face(frame)
                detect_ms = (time.time() - t_detect) * 1000
            except Exception as e:
                logging.error(f'Detection error: {e}')
                continue
            if faces is None:
                no_face_frames += 1
                continue  # no face in frame

            is_blurry = sharpness < BLUR_THRESHOLD
            forced = False
            if is_blurry:
                if (now - last_processed) * 1000 >= FORCE_AFTER_MS:
                    forced = True   # failsafe: don't starve recognition
                else:
                    skipped_blurry += 1
                    continue

            # --- Process the frame ---
            last_processed = now
            try:
                t0 = time.time()
                if in_fast:
                    emb = self._sface_embed(frame, faces)
                    match = self._match_sface(emb) if emb is not None else None
                    fast_ms += (time.time() - t0) * 1000 + detect_ms
                    fast_frames += 1
                else:
                    emb = self._get_heavy_embedding(frame)
                    match = self._match_heavy(emb) if emb is not None else None
                    heavy_ms += (time.time() - t0) * 1000 + detect_ms
                    heavy_frames += 1
            except Exception as e:
                logging.error(f'Recognition error: {e}')
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
            'fast_avg_ms':  round(fast_ms  / fast_frames,  1) if fast_frames  else None,
            'fast_frames':  fast_frames,
            'heavy_avg_ms': round(heavy_ms / heavy_frames, 1) if heavy_frames else None,
            'heavy_frames': heavy_frames,
            'forced_processed': forced_processed,
            'auto_processed': auto_processed,
            'skipped_blurry': skipped_blurry,
            'no_face_frames': no_face_frames,
            'duration_s':   duration_s,
        }

        # Persist calibration data (sharpness -> match outcome histogram)
        if self.blur_calibration is not None:
            try:
                self.blur_calibration.record_batch(
                    samples, forced_processed, auto_processed, skipped_blurry,
                    blur_threshold=BLUR_THRESHOLD, force_after_ms=FORCE_AFTER_MS)
            except Exception as e:
                logging.error(f'Failed to record blur calibration: {e}')

        if match:
            logging.info(
                f'[{match["model"]}] {match["name"]} {match["similarity"]*100:.1f}% — '
                f'sface {timing["fast_avg_ms"]}ms×{fast_frames}f  '
                f'heavy {timing["heavy_avg_ms"]}ms×{heavy_frames}f  '
                f'(forced {forced_processed}, auto {auto_processed}, skipped {skipped_blurry}, no_face {no_face_frames})'
            )
            if self.event_logger is not None:
                self.event_logger.log('face_recognized',
                                      name=match['name'],
                                      similarity=round(match['similarity'], 4),
                                      model=match['model'],
                                      snapshot=snapshot_filename,
                                      **timing)
            self._unlock_and_publish(match['name'])
            if match.get('migrate'):
                threading.Thread(
                    target=self._migrate_to_sface,
                    args=(match['name'], list(frame_buffer)),
                    daemon=True
                ).start()
        else:
            logging.warning(
                f'No face matched — '
                f'sface {timing["fast_avg_ms"]}ms×{fast_frames}f  '
                f'heavy {timing["heavy_avg_ms"]}ms×{heavy_frames}f  '
                f'(forced {forced_processed}, auto {auto_processed}, skipped {skipped_blurry}, no_face {no_face_frames})'
            )
            if self.event_logger is not None:
                self.event_logger.log('face_denied',
                                      similarity=None,
                                      snapshot=snapshot_filename,
                                      **timing)

    # ---------------------------------------------------------- benchmark

    def benchmark(self, iterations=20):
        """Measure raw per-model latency on live frames, as if a face were present.
        No blur gate, no early-out. Synthesizes a centered 112x112 crop for the
        SFace feature pass when no face is detected, so timing reflects the full
        pipeline regardless of who's in frame. Returns averages in ms."""
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

        buffalo, det_full, det_small, feat = [], [], [], []
        faces_detected = 0

        for f in frames:
            # buffalo_sc full pipeline (own detection on a 320x240 resize)
            t = time.time()
            self._heavy_model.get(cv2.resize(f, (320, 240)))
            buffalo.append((time.time() - t) * 1000)

            if not self._sface_ready:
                continue

            fh, fw = f.shape[:2]
            # YuNet detection at full resolution
            t = time.time()
            self._yunet.setInputSize((fw, fh))
            _, faces = self._yunet.detect(f)
            det_full.append((time.time() - t) * 1000)
            has_face = faces is not None and len(faces) > 0
            if has_face:
                faces_detected += 1

            # YuNet detection at 320 long side
            scale = 320.0 / max(fh, fw)
            small = cv2.resize(f, (round(fw * scale), round(fh * scale)))
            t = time.time()
            self._yunet.setInputSize((small.shape[1], small.shape[0]))
            self._yunet.detect(small)
            det_small.append((time.time() - t) * 1000)

            # SFace feature extraction (align real face if present, else center crop)
            t = time.time()
            if has_face:
                aligned = self._sface.alignCrop(f, faces[0])
            else:
                s = min(fh, fw)
                cy, cx = fh // 2, fw // 2
                crop = f[cy - s // 2:cy + s // 2, cx - s // 2:cx + s // 2]
                aligned = cv2.resize(crop, (112, 112))
            self._sface.feature(aligned)
            feat.append((time.time() - t) * 1000)

        result = {
            'frames': len(frames),
            'resolution': f'{w}x{h}',
            'faces_detected': faces_detected,
            'buffalo_ms': avg(buffalo),
            'sface_ready': self._sface_ready,
        }
        if self._sface_ready:
            df, ds, ft = avg(det_full), avg(det_small), avg(feat)
            result.update({
                'sface_detect_fullres_ms': df,
                'sface_detect_320_ms': ds,
                'sface_feature_ms': ft,
                'sface_total_fullres_ms': round((df or 0) + (ft or 0), 1),
                'sface_total_320_ms': round((ds or 0) + (ft or 0), 1),
            })
        logging.info(f'Benchmark: {result}')
        return result

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
