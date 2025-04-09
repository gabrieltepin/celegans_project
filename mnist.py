import os
import struct
import time
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

SEED = 11
np.random.seed(SEED)

def _load_idx_images(filepath: str):
    """Return images as (n, 28, 28) floats in [0,1]."""
    with open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "Magic number mismatch for images file."
        images = np.frombuffer(f.read(), dtype=np.uint8)
    return images.reshape(num, rows, cols).astype(np.float32) / 255.0


def _load_idx_labels(filepath: str):
    with open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, "Magic number mismatch for labels file."
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def standardise(X: np.ndarray, eps: float = 1e-8):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + eps
    return (X - mean) / std, mean, std


def one_hot(y: np.ndarray, num_classes: int = 10):
    return np.eye(num_classes, dtype=np.float32)[y]

def _hog_single(
    img: np.ndarray,
    *,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (4, 4),
    cells_per_block: Tuple[int, int] = (2, 2),
    clip: float = 0.2,
    eps: float = 1e-5,
):
    """Compute HOG descriptor for a single 28×28 image (float32, [0,1])."""
    h, w = img.shape
    cell_h, cell_w = pixels_per_cell
    n_cells_y, n_cells_x = h // cell_h, w // cell_w  # 7×7 for MNIST

    # --- Gradients (simple [-1,0,1] kernel) ---
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]

    magnitude = np.hypot(gx, gy)
    orientation = (np.rad2deg(np.arctan2(gy, gx)) % 180)  # unsigned [0,180)

    # --- Per‑cell histograms ---
    bin_width = 180 / orientations
    hist = np.zeros((n_cells_y, n_cells_x, orientations), dtype=np.float32)

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0, y1 = cy * cell_h, (cy + 1) * cell_h
            x0, x1 = cx * cell_w, (cx + 1) * cell_w
            cell_ori = orientation[y0:y1, x0:x1].ravel()
            cell_mag = magnitude[y0:y1, x0:x1].ravel()
            bins = (cell_ori // bin_width).astype(int)
            hist[cy, cx] = np.bincount(bins, weights=cell_mag, minlength=orientations)

    # --- Block normalisation (L2‑Hys) ---
    by, bx = cells_per_block
    blocks_y = n_cells_y - by + 1
    blocks_x = n_cells_x - bx + 1
    hog_vector = []

    for y in range(blocks_y):
        for x in range(blocks_x):
            block = hist[y : y + by, x : x + bx, :].ravel()
            norm = np.linalg.norm(block) + eps
            block = block / norm
            # Hys clipping
            block = np.clip(block, 0, clip)
            block = block / (np.linalg.norm(block) + eps)
            hog_vector.append(block)

    return np.concatenate(hog_vector)

def hog_batch(images: np.ndarray):
    return np.array([_hog_single(img) for img in images], dtype=np.float32)

def softmax(z: np.ndarray):
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z_shifted)
    return exp / exp.sum(axis=1, keepdims=True)

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10):
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

class SoftmaxSGD:
    def __init__(
        self,
        n_features: int,
        n_classes: int = 10,
        lr: float = 0.1,
        batch_size: int = 256,
        epochs: int = 40,
        lambda_reg: float = 1e-4,
        beta: float = 0.9,
        lr_decay: float = 0.95,
        seed: int = SEED,
    ):
        rng = np.random.default_rng(seed)
        self.W = 0.01 * rng.standard_normal((n_features, n_classes)).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.beta = beta
        self.lr_decay = lr_decay
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def _update_batch(self, Xb: np.ndarray, yb: np.ndarray):
        n = Xb.shape[0]
        probs = softmax(Xb @ self.W + self.b)
        grad = (probs - yb) / n
        gW = Xb.T @ grad + self.lambda_reg * self.W
        gb = grad.sum(axis=0)
        self.vW = self.beta * self.vW + (1 - self.beta) * gW
        self.vb = self.beta * self.vb + (1 - self.beta) * gb
        self.W -= self.lr * self.vW
        self.b -= self.lr * self.vb

    def fit(self, X: np.ndarray, y_oh: np.ndarray, verbose: bool = True):
        n_samples = X.shape[0]
        for epoch in range(1, self.epochs + 1):
            idx = np.random.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y_oh[idx]
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                self._update_batch(X_shuf[start:end], y_shuf[start:end])
            self.lr *= self.lr_decay
            if verbose and epoch % 5 == 0:
                loss = cross_entropy(y_oh, self.predict_proba(X)) + 0.5 * self.lambda_reg * np.sum(self.W * self.W)
                print(f"Epoch {epoch:02d}/{self.epochs} – loss: {loss:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def plot_confusion_matrix(cm, class_labels=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def main():
    t0 = time.time()
    
    # -------------------- Preprocess ------------------------
    MNIST_DIR = "MNIST"  # folder containing the 4 .idx files
    X_train_raw = _load_idx_images(os.path.join(MNIST_DIR, "train-images.idx3-ubyte"))
    y_train = _load_idx_labels(os.path.join(MNIST_DIR, "train-labels.idx1-ubyte"))
    X_test_raw = _load_idx_images(os.path.join(MNIST_DIR, "t10k-images.idx3-ubyte"))
    y_test = _load_idx_labels(os.path.join(MNIST_DIR, "t10k-labels.idx1-ubyte"))

    print("Extracting HOG features …")
    X_train_hog = hog_batch(X_train_raw)
    X_test_hog = hog_batch(X_test_raw)

    X_train_z, mean_feat, std_feat = standardise(X_train_hog)
    X_test_z = (X_test_hog - mean_feat) / std_feat

    y_train_oh = one_hot(y_train)
    print(f"Preprocessing time: {time.time() - t0:.1f}s")

    # -------------------- Train -----------------------------
    model = SoftmaxSGD(
        n_features=X_train_z.shape[1],
        lr=0.1,
        batch_size=256,
        epochs=40,
        lambda_reg=1e-4,
        beta=0.9,
        lr_decay=0.95,
    )
    model.fit(X_train_z, y_train_oh, verbose=True)
    print(f"Training time: {time.time() - t0:.1f}s")

    # -------------------- Evaluate --------------------------
    y_pred = model.predict(X_test_z)
    test_acc = (y_pred == y_test).mean()
    print(f"\nTest accuracy: {test_acc * 100:.2f}%")
    print(f"Evaluation time: {time.time() - t0:.1f}s")

    # -------------------- Save artefacts --------------------
    artefacts = {
        "W": model.W,
        "b": model.b,
        "mean_feat": mean_feat,
        "std_feat": std_feat,
        "hog_params": {
            "orientations": 9,
            "pixels_per_cell": (4, 4),
            "cells_per_block": (2, 2),
        },
    }
    with open("mnist_model.pkl", "wb") as f:
        pickle.dump(artefacts, f)
        
    cm = compute_confusion_matrix(y_test, y_pred, num_classes=10)
    plot_confusion_matrix(cm, class_labels=[str(i) for i in range(10)])

    print(f"Total wall‑clock time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
