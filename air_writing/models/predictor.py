"""
predictor.py — Real-time character predictor + AutoCompleter.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import utils.config as C
from utils.logger import get_logger
from utils.prediction_smoother import PredictionSmoother

log = get_logger(__name__, C.LOG_DIR, C.DEBUG_MODE)

_WORDS = [
    "the","be","to","of","and","a","in","that","have","it","for","not","on",
    "with","he","as","you","do","at","this","but","his","by","from","they","we",
    "say","her","she","or","an","will","my","one","all","would","there","their",
    "what","so","up","out","if","about","who","get","which","go","me","when",
    "make","can","like","time","no","just","him","know","take","people","into",
    "year","your","good","some","could","them","see","other","than","then","now",
    "look","only","come","its","over","think","also","back","after","use","two",
    "how","our","work","first","well","way","even","new","want","because","any",
    "hello","world","python","deep","learning","neural","network","hand","finger",
    "air","write","camera","model","predict","letter","word","text","image",
    "screen","gesture","detect","recognition","system","real","time","live",
]


class CharacterPredictor:
    """Load the CNN and predict characters in real time."""

    def __init__(self):
        self._model   = None
        self._labels  = C.LABELS
        self._smoother = PredictionSmoother()
        self._load()

    # ── public ───────────────────────────────────────────────────────────────

    def predict(self, image: np.ndarray,
                smooth: bool = True) -> Tuple[Optional[str], float]:
        """
        image: (28,28,1) float32 in [0,1]
        Returns (label, confidence). label=None if below threshold.
        """
        if self._model is None:
            return None, 0.0
        t0    = time.perf_counter()
        probs = self._model.predict(image[np.newaxis, ...], verbose=0)[0]
        ms    = (time.perf_counter() - t0) * 1000
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])
        label = self._labels[idx]
        log.debug("Pred=%s conf=%.2f (%.1fms)", label, conf, ms)
        if smooth:
            label, conf = self._smoother.update(label, conf)
        return label, conf

    def top_k(self, image: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        if self._model is None:
            return []
        probs = self._model.predict(image[np.newaxis, ...], verbose=0)[0]
        idxs  = np.argsort(probs)[::-1][:k]
        return [(self._labels[i], float(probs[i])) for i in idxs]

    def reset_smoother(self):
        self._smoother.reset()

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    # ── internal ─────────────────────────────────────────────────────────────

    def _load(self):
        try:
            import tensorflow as tf
            for g in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(g, True)
            if Path(C.MODEL_PATH).exists():
                self._model = tf.keras.models.load_model(C.MODEL_PATH)
                dummy = np.zeros((1, 28, 28, 1), dtype=np.float32)
                self._model.predict(dummy, verbose=0)   # warm-up
                log.info("Model loaded + warmed up: %s", C.MODEL_PATH)
            else:
                log.warning("Model not found at %s — run training/train.py first.", C.MODEL_PATH)
        except Exception as e:
            log.error("Model load failed: %s", e)


class AutoCompleter:
    """Prefix-based word suggestions."""
    def __init__(self, words=None):
        self._words = sorted(set(words or _WORDS))

    def suggest(self, prefix: str, n: int = 4) -> List[str]:
        p = prefix.lower()
        return [w for w in self._words if w.startswith(p)][:n]
