import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from PIL import Image
import pickle
import time

np.random.seed(11)

def load_idx_images(filepath):
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols).astype(np.float32) / 255.0
    return images

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

def summarize_mnist_dataset(X, y, dataset_name="MNIST"):
    num_images = X.shape[0]
    image_shape = X.shape[1:]
    unique_labels, label_counts = np.unique(y, return_counts=True)
    print(f"Dataset: {dataset_name}")
    print(f" - Number of images: {num_images}")
    print(f" - Image size (Height x Width): {image_shape}")
    print(f" - Classes: {unique_labels}")
    print(" - Class distribution:")
    for label, count in zip(unique_labels, label_counts):
        print(f"     Digit {label}: {count} samples")
    print()

def pca(X, n_components=50):
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec
    n_samples = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx_sorted = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx_sorted]
    # Keep top n_components
    pcs = eigenvectors[:, :n_components]
    X_pca = X_centered @ pcs
    return X_pca, pcs, mean_vec

def compute_pca(X, pcs, mean_vec):
    X_centered = X - mean_vec
    return X_centered @ pcs

def PHI(X, degree):
    X_expanded_list = [X**d for d in range(1, degree + 1)]
    X_expanded = np.concatenate(X_expanded_list, axis=1)
    return X_expanded

def normalize_features(X, eps=1e-8):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + eps
    X_norm = (X - mean) / std
    return X_norm, mean, std

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

def categorical_cross_entropy(y_true, y_pred):
    """
    computes the average cross-entropy for the batch.
    y_true is (n_samples, n_classes).
    y_pred is (n_samples, n_classes).
    """
    epsilon = 1e-10
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))

def train_softmax_regression(
    X, 
    y, 
    lr=1e-3, 
    epochs=100, 
    lambda_reg=0.0,   # L2 regularization coefficient
    beta=0.9,         # momentum hyperparameter
    verbose=True
):
    """
    X: (n_samples, n_features)
    y: (n_samples, n_classes) one-hot
    lr: learning rate
    epochs: number of training epochs
    lambda_reg: L2 regularization strength
    beta: momentum factor (0 means no momentum)
    """
    n_samples, n_features = X.shape
    n_classes = y.shape[1]

    # Initialize weights and biases
    W = np.zeros((n_features, n_classes), dtype=np.float32)
    b = np.zeros(n_classes, dtype=np.float32)

    # Initialize velocity (for momentum)
    vW = np.zeros_like(W)
    vb = np.zeros_like(b)

    for epoch in range(epochs):
        # Forward pass
        logits = X @ W + b  # shape (n_samples, n_classes)
        probs = softmax(logits)

        # Compute loss (cross entropy + L2 penalty on W)
        loss_ce = categorical_cross_entropy(y, probs)
        loss_reg = 0.5 * lambda_reg * np.sum(W * W)  # typical L2 term
        loss = loss_ce + loss_reg

        # Gradient wrt W, b (average over samples)
        grad_W = (1/n_samples) * X.T @ (probs - y)  
        grad_b = (1/n_samples) * np.sum(probs - y, axis=0)

        # Add L2 regularization to dW
        # (Note: often we do NOT regularize the bias term)
        grad_W += lambda_reg * W

        # Momentum updates
        # vW, vb accumulate gradients
        vW = beta * vW + (1 - beta) * grad_W
        vb = beta * vb + (1 - beta) * grad_b

        # Parameter update
        W -= lr * vW
        b -= lr * vb

        # Print info if verbose
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

    return W, b

#######################################################
def evaluate_softmax_model(X_train, y_train_oh, X_val, y_val, params, verbose=False):
    W, b = train_softmax_regression(
        X_train, y_train_oh,
        lr=params['lr'],
        epochs=params['epochs'],
        lambda_reg=params['lambda_reg'],
        beta=params['beta'],
        verbose=verbose
    )
    y_pred = predict(X_val, W, b)
    acc = np.mean(y_pred == y_val)
    return acc
def grid_search_softmax(X_train, y_train, y_train_oh, param_grid):
    param_names = list(param_grid.keys())
    best_acc = 0
    best_params = None

    print(f"\n🔍 Grid Search over {len(list(product(*param_grid.values())))} combinations...\n")

    for values in product(*param_grid.values()):
        params = dict(zip(param_names, values))

        acc = evaluate_softmax_model(X_train, y_train_oh, X_train, y_train, params)

        print(f"Params: {params} → Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_params = params

    print(f"\n✅ Best Params: {best_params} → Accuracy: {best_acc:.4f}")
    return best_params



## ------

from sklearn.model_selection import StratifiedKFold
def grid_search_softmax_cv(X, y, y_oh, param_grid, n_splits=5):
    param_names = list(param_grid.keys())
    best_avg_acc = 0
    best_params = None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"\n🔍 Grid Search with {n_splits}-fold CV over {len(list(product(*param_grid.values())))} combinations...\n")

    for values in product(*param_grid.values()):
        params = dict(zip(param_names, values))
        accuracies = []

        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold_oh = y_oh[train_idx]
            y_val_fold = y[val_idx]

            acc = evaluate_softmax_model(X_train_fold, y_train_fold_oh, X_val_fold, y_val_fold, params)
            accuracies.append(acc)

        avg_acc = np.mean(accuracies)
        print(f"Params: {params} → Avg Accuracy: {avg_acc:.4f}")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_params = params

    print(f"\n✅ Best Params: {best_params} → Avg Accuracy: {best_avg_acc:.4f}")
    return best_params

####################################################################

def predict(X, W, b):
    logits = X @ W + b
    probs = softmax(logits)
    return np.argmax(probs, axis=1)


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """
    y_true: (n_samples,) with class labels 0..9
    y_pred: (n_samples,) with class labels 0..9
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label][pred_label] += 1
    return cm

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("MNIST Confusion Matrix")
    plt.show()



if __name__ == "__main__":
    start_total_time = time.time()

    # ---------- Preprocessing ----------
    start_prep_time = time.time()
    mnist_dir = "MNIST"  # folder where the MNIST .idx files are stored
    X_train = load_idx_images(os.path.join(mnist_dir, "train-images.idx3-ubyte"))
    y_train = load_idx_labels(os.path.join(mnist_dir, "train-labels.idx1-ubyte"))
    X_test = load_idx_images(os.path.join(mnist_dir, "t10k-images.idx3-ubyte"))
    y_test = load_idx_labels(os.path.join(mnist_dir, "t10k-labels.idx1-ubyte"))
    # summarize_mnist_dataset(X_train, y_train, "MNIST Train")
    # summarize_mnist_dataset(X_test, y_test, "MNIST Test")

    # flatten each image to a 1D vector: shape => (n_samples, 28*28)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # n_components = 300
    n_components = 400
    X_train_pca, pcs, mean_vec = pca(X_train, n_components=n_components)
    X_test_pca = compute_pca(X_test, pcs, mean_vec)

    # feature engineering
    M=7
    X_train_pca = PHI(X_train_pca, M)
    X_test_pca  = PHI(X_test_pca, M)
    # from sklearn.kernel_approximation import RBFSampler
    # rbf = RBFSampler(gamma=0.05, n_components=2000, random_state=11)
    # X_train_pca = rbf.fit_transform(X_train_pca)
    # X_test_pca  = rbf.transform(X_test_pca)

    # normalization
    # X_train, mean_feat, std_feat = normalize_features(X_train)
    # X_test = (X_test - mean_feat) / std_feat
    X_train_norm, mean_feat, std_feat = normalize_features(X_train_pca)
    X_test_norm = (X_test_pca - mean_feat) / std_feat

    # converting labels to one-hot
    y_train_oh = one_hot(y_train, num_classes=10)
    y_test_oh = one_hot(y_test, num_classes=10)

    end_prep_time = time.time()
    print(f"\nPreprocessing Time: {end_prep_time - start_prep_time:.4f} seconds")

    # ---------- Training ----------
    start_exec_time = time.time()
    # W, b = train_softmax_regression(X_train, y_train_oh, lr=0.00025, epochs=20, verbose=True)
    # W, b = train_softmax_regression(X_train_norm, y_train_oh, lr=0.9, epochs=50, lambda_reg=0, beta=0.95, verbose=True)
    ##############################################
    # param_grid = {
    #     'lr': [0.9],
    #     'epochs': [100],
    #     'lambda_reg': [0.05],
    #     'beta': [0.8]
    # }

    # best_params = grid_search_softmax(X_train_norm, y_train, y_train_oh, param_grid)

    # # Train final model using best params
    # W, b = train_softmax_regression(
    #     X_train_norm, y_train_oh,
    #     lr=best_params['lr'],
    #     epochs=best_params['epochs'],
    #     lambda_reg=best_params['lambda_reg'],
    #     beta=best_params['beta'],
    #     verbose=True
    # )
    ###############################################
    # param_grid = {
    #     'lr': [0.9, 0.95],
    #     'epochs': [50],
    #     'lambda_reg': [0.05,],
    #     'beta': [0.8, 0.7]
    # }

    # best_params = grid_search_softmax_cv(X_train_norm, y_train, y_train_oh, param_grid, n_splits=5)

    # # Final training on full training set with best params
    # W, b = train_softmax_regression(
    #     X_train_norm, y_train_oh,
    #     lr=best_params['lr'],
    #     epochs=best_params['epochs'],
    #     lambda_reg=best_params['lambda_reg'],
    #     beta=best_params['beta'],
    #     verbose=True
    # )
    ###############################################
    param_grid = {
        'lr': 0.9,
        'epochs': 50,
        'lambda_reg': 0.05,
        'beta': 0.8,
    }
    W, b = train_softmax_regression(
        X_train_norm, y_train_oh,
        lr=param_grid['lr'],
        epochs=param_grid['epochs'],
        lambda_reg=param_grid['lambda_reg'],
        beta=param_grid['beta'],
        verbose=True
    )
    end_exec_time = time.time()
    print(f"\nTraining Time: {end_exec_time - start_exec_time:.4f} seconds")

    # ---------- Set the model ----------
    # y_pred = predict(X_test, W, b)
    y_pred = predict(X_test_norm, W, b)

    test_acc = np.mean(y_pred == y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    end_total_time = time.time()
    print(f"\nTotal Time: {end_total_time - start_total_time:.4f} seconds")

    # ---------- Saving the model ----------
    # w = train_softmax_regression(X_train_norm, )
    model_dict = {
        'w': W,
        'pcs': pcs,
        'mean_pca': mean_vec,
        'mean_train': mean_feat,
        'std_train': std_feat,
        'phi_degree': M
    }

    with open("mnist_model.pkl", "wb") as f:
        pickle.dump(model_dict, f)

    # # ---------- Results ----------
    # cm = compute_confusion_matrix(y_test, y_pred, num_classes=10)
    # plot_confusion_matrix(cm)

