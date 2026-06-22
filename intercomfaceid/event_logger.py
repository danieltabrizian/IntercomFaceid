import json
import os
import re
import shutil
import threading
import logging
import cv2
from datetime import datetime


class EventLogger:
    def __init__(self, data_dir='/data'):
        self.events_file = os.path.join(data_dir, 'events.jsonl')
        self.snapshots_dir = os.path.join(data_dir, 'snapshots')
        self.face_snapshots_dir = os.path.join(data_dir, 'face_snapshots')
        self._lock = threading.Lock()
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.face_snapshots_dir, exist_ok=True)
        logging.info("EventLogger initialized")

    def log(self, event_type, **kwargs):
        event = {'timestamp': datetime.now().isoformat(), 'type': event_type, **kwargs}
        with self._lock:
            with open(self.events_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        return event

    def get_recent(self, limit=200):
        if not os.path.exists(self.events_file):
            return []
        with self._lock:
            with open(self.events_file, 'r') as f:
                lines = f.readlines()
        events = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
            if len(events) >= limit:
                break
        return events

    def get_all(self):
        if not os.path.exists(self.events_file):
            return []
        with self._lock:
            with open(self.events_file, 'r') as f:
                content = f.read()
        events = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return events

    def save_snapshot(self, frame, prefix='event'):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'{prefix}_{ts}.jpg'
        path = os.path.join(self.snapshots_dir, filename)
        cv2.imwrite(path, frame)
        self._prune_snapshots()
        return filename

    def _prune_snapshots(self, max_keep=500):
        """Cap the snapshots dir so per-signal images can't fill the disk —
        keep the newest max_keep, delete the rest."""
        try:
            files = [os.path.join(self.snapshots_dir, f)
                     for f in os.listdir(self.snapshots_dir) if f.endswith('.jpg')]
            if len(files) <= max_keep:
                return
            files.sort(key=os.path.getmtime)
            for f in files[:len(files) - max_keep]:
                try:
                    os.remove(f)
                except OSError:
                    pass
        except OSError:
            pass

    # ---- per-person face image galleries ----
    # Each person gets a directory /data/face_snapshots/<name>/ holding the most
    # recent N face crops (newest kept, oldest pruned). Legacy single
    # /data/face_snapshots/<name>.jpg files are still served if no dir exists.

    def _safe(self, name):
        """Filesystem-safe folder name for a person."""
        return re.sub(r'[^A-Za-z0-9._ -]', '_', name).strip() or 'unknown'

    def _face_dir(self, name):
        return os.path.join(self.face_snapshots_dir, self._safe(name))

    def add_face_image(self, img, name, max_keep=6):
        """Save a face crop into the person's gallery, pruning to newest max_keep."""
        d = self._face_dir(name)
        os.makedirs(d, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        cv2.imwrite(os.path.join(d, f'{ts}.jpg'), img)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith('.jpg'))
        while len(files) > max_keep:
            try:
                os.remove(os.path.join(d, files.pop(0)))
            except OSError:
                break
        return f'{self._safe(name)}/{ts}.jpg'

    # back-compat alias used by enrollment
    def save_face_snapshot(self, frame, name):
        return self.add_face_image(frame, name)

    def face_images(self, name):
        """Web-relative image paths for a person, newest first."""
        d = self._face_dir(name)
        if os.path.isdir(d):
            files = sorted((f for f in os.listdir(d) if f.lower().endswith('.jpg')), reverse=True)
            if files:
                return [f'{self._safe(name)}/{f}' for f in files]
        legacy = os.path.join(self.face_snapshots_dir, f'{self._safe(name)}.jpg')
        if os.path.exists(legacy):
            return [f'{self._safe(name)}.jpg']
        return []

    def face_snapshot_exists(self, name):
        return len(self.face_images(name)) > 0

    def rename_face_images(self, old, new):
        od, nd = self._face_dir(old), self._face_dir(new)
        if os.path.isdir(od):
            if os.path.isdir(nd):
                # merge old images into the (possibly existing) new dir
                for f in os.listdir(od):
                    try:
                        os.replace(os.path.join(od, f), os.path.join(nd, f))
                    except OSError:
                        pass
                shutil.rmtree(od, ignore_errors=True)
            else:
                os.rename(od, nd)
        legacy = os.path.join(self.face_snapshots_dir, f'{self._safe(old)}.jpg')
        if os.path.exists(legacy):
            os.makedirs(nd, exist_ok=True)
            try:
                os.replace(legacy, os.path.join(nd, 'legacy.jpg'))
            except OSError:
                pass

    def delete_face_images(self, name):
        shutil.rmtree(self._face_dir(name), ignore_errors=True)
        legacy = os.path.join(self.face_snapshots_dir, f'{self._safe(name)}.jpg')
        if os.path.exists(legacy):
            try:
                os.remove(legacy)
            except OSError:
                pass
