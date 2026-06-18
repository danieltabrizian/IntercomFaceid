import json
import os
import threading
import logging
from datetime import datetime

BIN_WIDTH = 20      # sharpness (Laplacian variance) bucket size
MAX_BIN = 800       # values above this are lumped into the top bucket


class BlurCalibration:
    """Persistent histogram of face-crop sharpness vs. whether the frame matched.

    The whole point: we cannot pick a blur threshold by guessing. Instead we
    process (nearly) every detected face for a collection period, record its
    sharpness and whether it produced a recognition match, and accumulate that
    into a histogram. After enough data, matches will cluster above some
    sharpness value — that value is the threshold to set.
    """

    def __init__(self, path='/data/blur_calibration.json'):
        self.path = path
        self._lock = threading.Lock()
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    d = json.load(f)
                    d.setdefault('bins', {})
                    d.setdefault('forced_processed', 0)
                    d.setdefault('auto_processed', 0)
                    d.setdefault('skipped_blurry', 0)
                    d.setdefault('total_matched', 0)
                    return d
            except Exception as e:
                logging.warning(f'Could not load blur calibration: {e}')
        return {
            'bins': {},               # bin_floor(str) -> {'n': int, 'matched': int}
            'forced_processed': 0,    # processed despite being below threshold (failsafe)
            'auto_processed': 0,      # processed because at/above threshold
            'skipped_blurry': 0,      # below threshold AND failsafe didn't fire
            'total_matched': 0,
            'blur_threshold': None,
            'force_after_ms': None,
            'updated': None,
        }

    def _bin_key(self, sharpness):
        b = int(min(sharpness, MAX_BIN) // BIN_WIDTH) * BIN_WIDTH
        return str(b)

    def record_batch(self, samples, forced_processed, auto_processed,
                     skipped_blurry, blur_threshold=None, force_after_ms=None):
        """samples: list of (sharpness, matched_bool) for processed frames."""
        if not samples and not skipped_blurry:
            return
        with self._lock:
            bins = self.data['bins']
            for sharp, matched in samples:
                k = self._bin_key(sharp)
                if k not in bins:
                    bins[k] = {'n': 0, 'matched': 0}
                bins[k]['n'] += 1
                if matched:
                    bins[k]['matched'] += 1
                    self.data['total_matched'] += 1
            self.data['forced_processed'] += forced_processed
            self.data['auto_processed'] += auto_processed
            self.data['skipped_blurry'] += skipped_blurry
            if blur_threshold is not None:
                self.data['blur_threshold'] = blur_threshold
            if force_after_ms is not None:
                self.data['force_after_ms'] = force_after_ms
            self.data['updated'] = datetime.now().isoformat()
            self._save_locked()

    def _save_locked(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)

    def summary(self):
        with self._lock:
            data = json.loads(json.dumps(self.data))  # deep copy

        bins = data['bins']
        floors = sorted(int(k) for k in bins.keys())
        hist = [
            {'floor': fl, 'n': bins[str(fl)]['n'], 'matched': bins[str(fl)]['matched']}
            for fl in floors
        ]

        # Suggested threshold: the highest bin floor below which NO match has ever
        # occurred (i.e. everything below this is safe to discard).
        first_match_floor = None
        for h in hist:
            if h['matched'] > 0:
                first_match_floor = h['floor']
                break

        # Stricter: 5th-percentile of matched sharpness (tolerates a rare low outlier).
        matched_floors = []
        for h in hist:
            matched_floors.extend([h['floor']] * h['matched'])
        p5_floor = None
        if matched_floors:
            idx = max(0, int(len(matched_floors) * 0.05) - 1)
            p5_floor = sorted(matched_floors)[idx]

        total_processed = data['forced_processed'] + data['auto_processed']
        return {
            'histogram': hist,
            'forced_processed': data['forced_processed'],
            'auto_processed': data['auto_processed'],
            'skipped_blurry': data['skipped_blurry'],
            'total_processed': total_processed,
            'total_matched': data['total_matched'],
            'forced_pct': round(100 * data['forced_processed'] / total_processed, 1) if total_processed else 0,
            'suggested_threshold_safe': first_match_floor,   # below this: zero matches ever
            'suggested_threshold_p5': p5_floor,              # below this: <5% of matches
            'current_blur_threshold': data['blur_threshold'],
            'current_force_after_ms': data['force_after_ms'],
            'bin_width': BIN_WIDTH,
            'updated': data['updated'],
        }
