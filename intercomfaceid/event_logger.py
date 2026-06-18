import json
import os
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
        return filename

    def save_face_snapshot(self, frame, name):
        filename = f'{name}.jpg'
        path = os.path.join(self.face_snapshots_dir, filename)
        cv2.imwrite(path, frame)
        return filename

    def face_snapshot_exists(self, name):
        return os.path.exists(os.path.join(self.face_snapshots_dir, f'{name}.jpg'))
