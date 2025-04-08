import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import struct
import time
import seaborn as sns
import pandas as pd

np.random.seed(11)

def get_celegans_image_paths(base_path: str, classes: list[str]) -> dict:
    #TODO: check if 5 labels are correct
    image_paths = {}
    for class_label in classes:
        class_dir = os.path.join(base_path, class_label)
        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.png')]
        image_paths[class_label] = images
    return image_paths

def display_celegans_sample_img(image_paths: dict, classes: list[str]):
    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes] 
    for ax, class_label in zip(axes, classes):
        img = Image.open(image_paths[class_label][0])
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Class {class_label}")
        ax.axis('off')
    plt.suptitle("Sample Celegans Images")
    plt.tight_layout()
    plt.show()

def summarize_celegans_dataset(image_paths: dict):
    for class_label, paths in image_paths.items():
        sample_img = Image.open(paths[0])
        print(f"Class {class_label}:")
        print(f" - Number of images: {len(paths)}")
        print(f" - Image size (Width x Height): {sample_img.size}")
        print(f" - Mode (e.g., RGB, L): {sample_img.mode}\n")

def load_idx_images(filepath):
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num_images, rows, cols) / 255.0

def load_idx_labels(filepath):
    with open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def visualize_mnist_images(images, labels, num_images=10):
    plt.figure(figsize=(num_images, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.suptitle("Sample MNIST Images")
    plt.show()

def summarize_mnist_dataset(X, y, dataset_name="MNIST Train"):
    num_images = X.shape[0]
    image_shape = X.shape[1:]
    unique_labels, label_counts = np.unique(y, return_counts=True)

    print(f"Dataset: {dataset_name}")
    print(f" - Number of images: {num_images}")
    print(f" - Image size (Height x Width): {image_shape}")
    print(f" - Total classes: {len(unique_labels)}")
    print(f" - Mode: Grayscale (float values in [0, 1])")
    print(" - Class distribution:")
    for label, count in zip(unique_labels, label_counts):
        print(f"     Digit {label}: {count} samples")
    print()

def summarize_mnist_dataset_table(X, y, dataset_name="MNIST Train"):
    num_images = X.shape[0]
    image_shape = X.shape[1:]
    height, width = image_shape
    unique_labels, label_counts = np.unique(y, return_counts=True)

    summary = pd.DataFrame(index=["Number of images", "Image height", "Image width", "Mode (grayscale)"])
    for label in range(10):
        count = label_counts[unique_labels.tolist().index(label)] if label in unique_labels else 0
        summary[label] = [count, height, width, "L"]

    print(f"Dataset: {dataset_name}")
    display(summary)

def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)  # stability trick
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # FIXED LINE

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-10
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))

def train_softmax_regression(X, y, lr=1e-3, epochs=100):
    #TODO: apply a rolling average to the 
    n_samples, n_features = X.shape
    n_classes = y.shape[1]
    W = np.zeros((n_features, n_classes))
    b = np.zeros(n_classes)

    for epoch in range(epochs):
        logits = X @ W + b
        probs = softmax(logits)
        loss = categorical_cross_entropy(y, probs)

        grad_W = (1/n_samples) * X.T @ (probs - y)
        grad_b = (1/n_samples) * np.sum(probs - y, axis=0)

        W -= lr * grad_W
        b -= lr * grad_b

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
    
    return W, b

def predict(X, W, b):
    probs = softmax(X @ W + b)
    return np.argmax(probs, axis=1)

def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def plot_confusion(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# ==== MAIN ====

mnist_dir = "MNIST"
# TODO: increase the X complexity by using the Phi(X) matrix
X_train = load_idx_images(os.path.join(mnist_dir, "train-images.idx3-ubyte"))
y_train = load_idx_labels(os.path.join(mnist_dir, "train-labels.idx1-ubyte"))
X_test = load_idx_images(os.path.join(mnist_dir, "t10k-images.idx3-ubyte"))
y_test = load_idx_labels(os.path.join(mnist_dir, "t10k-labels.idx1-ubyte"))

# Flatten images for training
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

# Train model
W, b = train_softmax_regression(X_train, y_train_oh, lr=1e-3, epochs=100)

# Predict
y_pred = predict(X_test, W, b)

# Show predictions
print("Predicted labels (first 20):", y_pred[:20])

# Evaluation (uncomment if desired)
acc = np.mean(y_pred == y_test)
print(f"Test Accuracy: {acc:.4f}")

cm = compute_confusion_matrix(y_test, y_pred)
plot_confusion(cm)
