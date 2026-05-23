"""
train.py — Train the Air Writing CNN.

Run from project root:
    python training/train.py
    python training/train.py --epochs 30 --batch 128

Outputs:
  models/air_writing_cnn.keras     ← saved model
  logs/training_curves.png         ← accuracy + loss graphs
  logs/confusion_matrix.png        ← per-class confusion
  logs/training_log.csv            ← epoch-by-epoch metrics
"""
from __future__ import annotations
import argparse, os, sys

# make sure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils.config as C
from utils.logger import get_logger
from training.dataset_loader import load_emnist, build_tf_datasets
from models.cnn_model import build_model, save_model, get_callbacks

log = get_logger("train", C.LOG_DIR, C.DEBUG_MODE)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",   type=int,   default=C.EPOCHS)
    p.add_argument("--batch",    type=int,   default=C.BATCH_SIZE)
    p.add_argument("--lr",       type=float, default=C.LEARNING_RATE)
    p.add_argument("--data-dir", default="datasets")
    p.add_argument("--debug",    action="store_true")
    return p.parse_args()


def plot_history(h):
    os.makedirs("logs", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d0d0d")
    for a in ax:
        a.set_facecolor("#1a1a2e")
        a.tick_params(colors="w")
        a.xaxis.label.set_color("w")
        a.yaxis.label.set_color("w")
        a.title.set_color("w")
        for s in a.spines.values():
            s.set_edgecolor("#333")

    ax[0].plot(h.history["accuracy"],     "#00d4ff", lw=2, label="Train")
    ax[0].plot(h.history["val_accuracy"], "#ff6b6b", lw=2, label="Val")
    ax[0].set(title="Accuracy", xlabel="Epoch", ylabel="Acc")
    ax[0].legend(facecolor="#1a1a2e", labelcolor="w")

    ax[1].plot(h.history["loss"],     "#00d4ff", lw=2, label="Train")
    ax[1].plot(h.history["val_loss"], "#ff6b6b", lw=2, label="Val")
    ax[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax[1].legend(facecolor="#1a1a2e", labelcolor="w")

    plt.tight_layout()
    plt.savefig("logs/training_curves.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    log.info("Curves → logs/training_curves.png")
    plt.close()


def plot_cm(y_true, y_pred, labels, top=36):
    from sklearn.metrics import confusion_matrix
    mask = (y_true < top) & (y_pred < top)
    cm   = confusion_matrix(y_true[mask], y_pred[mask],
                             labels=list(range(top)))
    cmn  = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(18, 16))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    im = ax.imshow(cmn, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(top)); ax.set_yticks(range(top))
    ax.set_xticklabels(labels[:top], rotation=90, fontsize=8, color="w")
    ax.set_yticklabels(labels[:top], fontsize=8, color="w")
    ax.set_xlabel("Predicted", color="w")
    ax.set_ylabel("True",      color="w")
    ax.set_title("Confusion Matrix (first 36 classes)", color="w")
    plt.tight_layout()
    plt.savefig("logs/confusion_matrix.png", dpi=120,
                bbox_inches="tight", facecolor="#0d0d0d")
    log.info("Confusion matrix → logs/confusion_matrix.png")
    plt.close()


def main():
    a = args()
    C.EPOCHS        = a.epochs
    C.BATCH_SIZE    = a.batch
    C.LEARNING_RATE = a.lr
    C.DEBUG_MODE    = a.debug

    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    log.info("GPU: %s", tf.config.list_physical_devices("GPU") or "none — CPU")

    (x_tr, y_tr), (x_te, y_te) = load_emnist(a.data_dir)
    train_ds, val_ds, test_ds   = build_tf_datasets(x_tr, y_tr, x_te, y_te)

    model = build_model()
    model.summary(print_fn=log.info)

    log.info("Training %d epochs …", a.epochs)
    os.makedirs("logs", exist_ok=True)
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=a.epochs, callbacks=get_callbacks(), verbose=1,
    )

    loss, acc = model.evaluate(test_ds, verbose=0)
    log.info("TEST  acc=%.4f  loss=%.4f", acc, loss)

    save_model(model)
    plot_history(history)
    preds = model.predict(x_te[:5000], verbose=0).argmax(1)
    plot_cm(y_te[:5000], preds, C.LABELS)
    log.info("All done. Model → %s", C.MODEL_PATH)


if __name__ == "__main__":
    main()
