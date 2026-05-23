"""prediction_smoother.py — Sliding-window majority-vote smoother."""
from collections import deque, Counter
from typing import Optional, Tuple
import utils.config as C


class PredictionSmoother:
    def __init__(self):
        self._win   = deque(maxlen=C.SMOOTH_WINDOW)
        self._thr   = C.CONF_THRESHOLD

    def update(self, label: str, conf: float) -> Tuple[Optional[str], float]:
        if conf >= self._thr:
            self._win.append(label)
        if not self._win:
            return None, 0.0
        top, cnt = Counter(self._win).most_common(1)[0]
        sc = cnt / len(self._win)
        return (top, sc) if sc >= self._thr else (None, sc)

    def reset(self):
        self._win.clear()
