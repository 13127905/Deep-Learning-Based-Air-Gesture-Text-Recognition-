"""
cnn_model.py — Deep CNN for 62-class character recognition.

Architecture  (28×28×1 → 62 classes)
─────────────────────────────────────
Block1: Conv32×2  BN ReLU MaxPool  →  14×14
Block2: Conv64×2  BN ReLU MaxPool  →   7×7
Block3: Conv128×2 BN ReLU MaxPool  →   3×3
Block4: Conv256   BN ReLU
Head  : GlobalAvgPool → Dense512 BN Drop0.4 → Dense256 BN Drop0.3 → Softmax62
"""
from __future__ import annotations
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import utils.config as C
from utils.logger import get_logger

log = get_logger(__name__, C.LOG_DIR, C.DEBUG_MODE)


def build_model() -> keras.Model:
    inp = keras.Input(shape=C.INPUT_SHAPE, name="img")

    def conv_block(x, f, pool=True):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if pool:
            x = layers.MaxPooling2D()(x)
        return x

    x = conv_block(inp,  32)
    x = conv_block(x,    64)
    x = conv_block(x,   128)
    # Block4: single conv, no pool
    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(C.DROPOUT_RATE)(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(C.NUM_CLASSES, activation="softmax", name="pred")(x)

    model = keras.Model(inp, out, name="AirWritingCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(C.LEARNING_RATE),
       loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    log.info("CNN built — %d params.", model.count_params())
    return model


def load_model(path: str = C.MODEL_PATH) -> keras.Model | None:
    if not Path(path).exists():
        log.warning("No model at %s.", path)
        return None
    log.info("Loading model from %s", path)
    return keras.models.load_model(path)


def save_model(model: keras.Model, path: str = C.MODEL_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    log.info("Model saved → %s", path)


def get_callbacks():
    Path(C.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            os.path.join(C.CHECKPOINT_DIR, "best.weights.h5"),
            monitor="val_accuracy", save_best_only=True,
            save_weights_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.CSVLogger("logs/training_log.csv"),
    ]
