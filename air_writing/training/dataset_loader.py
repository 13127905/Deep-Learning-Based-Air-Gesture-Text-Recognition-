"""
dataset_loader.py
Downloads EMNIST ByClass.  Falls back to synthetic data if offline.
"""
from __future__ import annotations
import gzip, os, struct, urllib.request
from pathlib import Path
from typing import Tuple
import numpy as np
import utils.config as C
from utils.logger import get_logger

log = get_logger(__name__, C.LOG_DIR, C.DEBUG_MODE)

_BASE  = "https://biometrics.nist.gov/cs_links/EMNIST/"
_FILES = {
    "train_img": "emnist-byclass-train-images-idx3-ubyte.gz",
    "train_lbl": "emnist-byclass-train-labels-idx1-ubyte.gz",
    "test_img":  "emnist-byclass-test-images-idx3-ubyte.gz",
    "test_lbl":  "emnist-byclass-test-labels-idx1-ubyte.gz",
}


def _dl(url, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s …", dest.name)
    urllib.request.urlretrieve(url, str(dest))


def _read_imgs(p):
    with gzip.open(str(p), "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)


def _read_lbls(p):
    with gzip.open(str(p), "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def _fix(imgs):
    """EMNIST is stored transposed+flipped."""
    return np.flip(np.transpose(imgs, (0, 2, 1)), axis=2)


def _synthetic(n=280):
    """Render chars with OpenCV when EMNIST is unavailable."""
    import cv2
    imgs, lbls = [], []
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX]
    rng = np.random.default_rng(42)
    for idx, ch in enumerate(C.LABELS):
        for _ in range(n):
            img   = np.zeros((28, 28), dtype=np.uint8)
            font  = fonts[rng.integers(len(fonts))]
            scale = 0.7 + rng.uniform(-0.15, 0.15)
            thick = int(rng.integers(1, 3))
            tx    = int(rng.integers(3, 9))
            ty    = int(rng.integers(18, 24))
            cv2.putText(img, ch, (tx, ty), font, scale, 200, thick, cv2.LINE_AA)
            noise = rng.integers(0, 28, img.shape, dtype=np.uint8)
            img   = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
            imgs.append(img); lbls.append(idx)
    return np.array(imgs), np.array(lbls, dtype=np.uint8)


def load_emnist(data_dir="datasets"):
    """Return ((x_tr,y_tr),(x_te,y_te)) — float32 (N,28,28,1) in [0,1]."""
    base = Path(data_dir)
    try:
        paths = {}
        for k, fn in _FILES.items():
            dest = base / fn
            _dl(_BASE + fn, dest)
            paths[k] = dest

        x_tr = _fix(_read_imgs(paths["train_img"]))
        y_tr = _read_lbls(paths["train_lbl"])
        x_te = _fix(_read_imgs(paths["test_img"]))
        y_te = _read_lbls(paths["test_lbl"])

        # keep only classes 0-61
        for _ in range(1):
            pass
        m_tr = y_tr < 62;  m_te = y_te < 62
        x_tr, y_tr = x_tr[m_tr], y_tr[m_tr]
        x_te, y_te = x_te[m_te], y_te[m_te]
        log.info("EMNIST: train=%d  test=%d", len(x_tr), len(x_te))

    except Exception as e:
        log.warning("EMNIST unavailable (%s) → synthetic data.", e)
        x_tr, y_tr = _synthetic(320)
        x_te, y_te = _synthetic(70)

    def prep(x):
        x = x.astype(np.float32) / 255.0
        return x[..., np.newaxis] if x.ndim == 3 else x

    return (prep(x_tr), y_tr), (prep(x_te), y_te)


def build_tf_datasets(x_tr, y_tr, x_te, y_te):
    import tensorflow as tf
    AUTO = tf.data.AUTOTUNE

    def aug(img, lbl):
        return tf.clip_by_value(img, 0.0, 1.0), lbl

    train_ds = (tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
                .shuffle(min(len(x_tr), 20_000))
                .map(aug, num_parallel_calls=AUTO)
                .batch(C.BATCH_SIZE).prefetch(AUTO))
    mid = len(x_te) // 2
    val_ds  = (tf.data.Dataset.from_tensor_slices((x_te[:mid], y_te[:mid]))
               .batch(C.BATCH_SIZE).prefetch(AUTO))
    test_ds = (tf.data.Dataset.from_tensor_slices((x_te[mid:], y_te[mid:]))
               .batch(C.BATCH_SIZE).prefetch(AUTO))
    return train_ds, val_ds, test_ds
