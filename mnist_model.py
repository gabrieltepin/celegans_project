import os
import pickle
import struct
import numpy as np
import pandas as pd
from typing import Tuple
from PIL import Image

def _load_idx_images(filepath: str):
    """Loads .idx3-ubyte MNIST format images."""
    with open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "Invalid magic number."
        images = np.frombuffer(f.read(), dtype=np.uint8)
    return images.reshape(num, rows, cols).astype(np.float32) / 255.0

def load_tif_images_from_folder(folder, target_size=(28, 28)):
    images = []
    names = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".tif"):
            path = os.path.join(folder, fname)
            img = Image.open(path).convert("L")
            img = img.resize(target_size)
            img_array = np.array(img).astype(np.float32) / 255.0
            images.append(img_array)
            names.append(fname)
    return np.array(images), names

def _hog_single(
    img: np.ndarray,
    *,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (4, 4),
    cells_per_block: Tuple[int, int] = (2, 2),
    clip: float = 0.2,
    eps: float = 1e-5,
):
    h, w = img.shape
    cell_h, cell_w = pixels_per_cell
    n_cells_y, n_cells_x = h // cell_h, w // cell_w

    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]

    magnitude = np.hypot(gx, gy)
    orientation = (np.rad2deg(np.arctan2(gy, gx)) % 180)

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

    by, bx = cells_per_block
    blocks_y = n_cells_y - by + 1
    blocks_x = n_cells_x - bx + 1
    hog_vector = []
    for y in range(blocks_y):
        for x in range(blocks_x):
            block = hist[y:y+by, x:x+bx, :].ravel()
            norm = np.linalg.norm(block) + eps
            block = block / norm
            block = np.clip(block, 0, clip)
            block = block / (np.linalg.norm(block) + eps)
            hog_vector.append(block)

    return np.concatenate(hog_vector)

def hog_batch(images: np.ndarray, **hog_params):
    return np.array([_hog_single(img, **hog_params) for img in images], dtype=np.float32)

def softmax(z: np.ndarray):
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z_shifted)
    return exp / exp.sum(axis=1, keepdims=True)

def predict(X: np.ndarray, W: np.ndarray, b: np.ndarray):
    probs = softmax(X @ W + b)
    return np.argmax(probs, axis=1)

if __name__ == "__main__":
    # --- Load model ---
    with open("mnist_model.pkl", "rb") as f:
        model = pickle.load(f)

    W = model["W"]
    b = model["b"]
    mean_feat = model["mean_feat"]
    std_feat = model["std_feat"]
    hog_params = model["hog_params"]

    # --- Ask user for test file ---
    base_path = input("Enter the path to the mnist files to classify: ").strip()
    while not os.path.isdir(base_path):
        print("That path doesn't exist. Try again.")
        base_path = input("Enter the path to the mnist files to classify: ").strip()
    # base_path = "test_mnist"  # folder with new .png images

    # --- Load and preprocess images ---
    # images = _load_idx_images(base_path)
    images, names = load_tif_images_from_folder(base_path)
    # X_hog = hog_batch(images, **hog_params)
    # X_norm = (X_hog - mean_feat) / std_feat

    # # --- Predict ---
    # y_pred = predict(X_norm, W, b)

    # # --- Output predictions ---
    # base_names = [f"img_{i}.png" for i in range(len(y_pred))]
    # df = pd.DataFrame({
    #     "image_name": base_names,
    #     "label": y_pred
    # })

    # # Add totals per class
    # value_counts = df["label"].value_counts().sort_index()
    # for i in range(10):
    #     total = value_counts.get(i, 0)
    #     df.loc[len(df)] = [f"TOTAL label {i}", total]

    # # Save results
    # output_file = "mnist_results.xlsx"
    # df.to_excel(output_file, index=False)

    # print(f"\nClassification complete. Results saved to {output_file}")
    if len(images) == 0:
        print("No .tif images found. Exiting.")
        exit(0)

    X_hog = hog_batch(images, **hog_params)
    X_norm = (X_hog - mean_feat) / std_feat

    # --- Predict ---
    y_pred = predict(X_norm, W, b)

    # --- Output predictions ---
    df = pd.DataFrame({
        "image_name": names,
        "label": y_pred
    })

    # Add totals per class
    value_counts = df["label"].value_counts().sort_index()
    for i in range(10):
        total = value_counts.get(i, 0)
        df.loc[len(df)] = [f"TOTAL label {i}", total]

    # Save results
    output_file = "mnist_results.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\nClassification complete. Results saved to {output_file}")
