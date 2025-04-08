import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import time
import seaborn as sns
import pandas as pd

np.random.seed(11)

def get_celegans_image_paths(base_path: str, classes: list[str]) -> dict:
    """Return a dictionary of image paths for each class."""
    image_paths = {}
    for class_label in classes:
        class_dir = os.path.join(base_path, class_label)
        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.png')]
        image_paths[class_label] = images
    return image_paths
    

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w):
    return sigmoid(X @ w)

def predict_class(X, w, threshold=0.5):
    probs = predict_proba(X, w)
    return (probs >= threshold).astype(int)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-10
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

def compute_gradients(X, y_true, y_pred):
    n = X.shape[0]
    error = y_pred - y_true
    grad_w = (1/n) * X.T @ error
    return grad_w

def load_celegans_design_matrix(image_paths: dict, img_size=(25, 25), max_per_class=None):
    X = []
    y = []
    for class_label, paths in image_paths.items():
        label = int(class_label)
        # limit images per class if specified
        if max_per_class:
            paths = paths[:max_per_class]
        for img_path in paths:
            img = Image.open(img_path).convert('L')
            img = img.resize(img_size)
            img_array = np.array(img).astype(np.float32) / 255.0
            X.append(img_array.flatten())
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    X = np.hstack((X, np.ones((X.shape[0], 1))))  # Add bias term as last feature
    return X, y

def pca(X, n_components=50):
    bias = X[:, -1].reshape(-1, 1)      
    X_no_bias = X[:, :-1]              
    mean_vec = np.mean(X_no_bias, axis=0)    
    X_centered = X_no_bias - mean_vec

    n_amostras = X_centered.shape[0]
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_amostras - 1)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    idx_sorted = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx_sorted]
    eigenvectors_sorted = eigenvectors[:, idx_sorted]

    principal_components = eigenvectors_sorted[:, :n_components]  

    X_reduced = np.dot(X_centered, principal_components)          
    X_pca = np.hstack((X_reduced, bias))   # shape (n_amostras, n_components + 1)

    return X_pca, principal_components, mean_vec

def train_logistic_regression_sgd(X, y, lr=1e-2, epochs=50, batch_size=32, beta=0.9, lambda_reg=0.01, verbose=True):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    v = np.zeros(n_features)

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            y_pred = predict_proba(X_batch, w)
            grad_w = compute_gradients(X_batch, y_batch, y_pred)

            # Add L2 regularization to gradient (but exclude bias term)
            grad_w[:-1] += lambda_reg * w[:-1]

            # EWMA momentum
            v = beta * v + (1 - beta) * grad_w
            w = w - lr * v

        if verbose and (epoch + 1) % 5 == 0:
            y_epoch_pred = predict_proba(X, w)
            loss = binary_cross_entropy(y, y_epoch_pred)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.6f}")

    return w


def normalize_features(X):
    X_features = X[:, :-1]  # exclude bias
    mean = np.mean(X_features, axis=0)
    std = np.std(X_features, axis=0) + 1e-8  # prevent division by zero
    X_norm = (X_features - mean) / std
    X_norm = np.hstack((X_norm, X[:, -1:]))  # reattach bias
    return X_norm, mean, std

def PHI(X, m):
    X_ = X[:, :-1]  # remove last column (bias)
    bias = X[:, -1:]        # save bias to reattach

    features = [X_**d for d in range(1, m + 1)]
    X = np.concatenate(features, axis=1)
    X = np.hstack((X, bias))  # reattach bias column
    return X

from itertools import product
def grid_search_logistic_regression(X, y, param_grid, test_size=0.1):
    from sklearn.model_selection import train_test_split

    param_names = list(param_grid.keys())
    best_accuracy = 0
    best_params = None

    print(f"\nStarting Grid Search over {len(list(product(*param_grid.values())))} combinations...\n")

    for values in product(*param_grid.values()):
        params = dict(zip(param_names, values))

        # Preprocessing
        X_phi = PHI(X, params['degree'])
        X_phi, _, _ = normalize_features(X_phi)

        X_train, X_test, y_train, y_test = train_test_split(X_phi, y, test_size=test_size, stratify=y)

        # Train
        w = train_logistic_regression_sgd(
            X_train, y_train,
            lr=params['lr'],
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            beta=params['beta'],
            lambda_reg=params['lambda_reg'],
            verbose=False
        )

        # Evaluate
        y_pred = predict_class(X_test, w)
        acc = np.mean(y_pred == y_test)

        print(f"Params: {params} → Accuracy: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_params = params

    print(f"\n✅ Best Params: {best_params} → Accuracy: {best_accuracy:.4f}")
    return best_params, best_accuracy

from sklearn.model_selection import StratifiedKFold
def grid_search_logistic_regression_cv(X, y, param_grid, n_splits=5):
    """
    Perform grid search using stratified k-fold cross-validation.
    param_grid: dict of {param_name: list of values}
    Returns: best_params, best_avg_accuracy
    """
    param_names = list(param_grid.keys())
    best_avg_accuracy = 0
    best_params = None

    total_combos = len(list(product(*param_grid.values())))
    print(f"\n🔍 Starting Grid Search with {n_splits}-Fold CV over {total_combos} combinations...\n")

    for i, values in enumerate(product(*param_grid.values())):
        params = dict(zip(param_names, values))
        print(f"🔁 [{i+1}/{total_combos}] Testing: {params}")

        # Preprocess for this configuration
        X_phi = PHI(X, params['degree'])
        X_phi, _, _ = normalize_features(X_phi)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracies = []

        for train_index, val_index in skf.split(X_phi, y):
            X_train, X_val = X_phi[train_index], X_phi[val_index]
            y_train, y_val = y[train_index], y[val_index]

            w = train_logistic_regression_sgd(
                X_train, y_train,
                lr=params['lr'],
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                beta=params['beta'],
                lambda_reg=params['lambda_reg'],
                verbose=False
            )

            y_pred = predict_class(X_val, w)
            acc = np.mean(y_pred == y_val)
            accuracies.append(acc)

        avg_acc = np.mean(accuracies)
        print(f"→ Avg Accuracy: {avg_acc:.4f}\n")

        if avg_acc > best_avg_accuracy:
            best_avg_accuracy = avg_acc
            best_params = params

    print(f"\n✅ Best Params: {best_params} → Avg Accuracy: {best_avg_accuracy:.4f}")
    return best_params, best_avg_accuracy


if __name__ == "__main__": 
    start_total_time = time.time()
    start_time = time.time()
    base_path = "Celegans_ModelGen"
    class_labels = ['0', '1']
    image_paths = get_celegans_image_paths(base_path, class_labels)

    # TODO: we should implement PHI matrix instead of directly using the linear input itself
    X, y = load_celegans_design_matrix(image_paths, img_size=(28, 28))
    # X, y = load_celegans_design_matrix(image_paths, img_size=(28, 28), max_per_class=200)
    
    # X = PHI(X,2)
    # X = PHI(X,1)
    X, mean, std = normalize_features(X)
    X, pcs, mean_pca = pca(X, n_components=500)

    #TODO: implement our own model selection code?
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
    end_time = time.time()
    print(f"\n🕒 Preprocessing Time: {end_time - start_time:.4f} seconds")

    # start_time = time.time()
    # w = train_logistic_regression_sgd(X_train, y_train, lr=1e-2, epochs=50, batch_size=100)
    # end_time = time.time()
    # start_time = time.time()
    # w = train_logistic_regression_sgd(
    #     X_train, y_train,
    #     lr=0.00025,
    #     epochs=25,
    #     batch_size=25,
    #     beta=0.95,
    #     lambda_reg=0.1
    # )
    # end_time = time.time()
    start_time = time.time()
    param_grid = {
        # 'lr': [0.00025, 0.005, 0.000025],
        'lr': [0.00025],
        'lambda_reg': [0.1],
        'beta': [0.95],
        'batch_size': [25],
        'epochs': [25],
        'degree': [25]
    }
    best_params, best_acc = grid_search_logistic_regression(X, y, param_grid)
    # best_params, best_acc = grid_search_logistic_regression_cv(X, y, param_grid, n_splits=5)
    end_time = time.time()
    #### ✅ Best Params: {'lr': 0.001, 'lambda_reg': 0.1, 'beta': 0.9, 'batch_size': 64, 'epochs': 50, 'degree': 3} → Accuracy: 0.8109
    #### ✅ Best Params: {'lr': 0.005, 'lambda_reg': 0.1, 'beta': 0.97, 'batch_size': 25, 'epochs': 50, 'degree': 2} → Accuracy: 0.8136
    #### ✅ Best Params: {'lr': 0.005, 'lambda_reg': 0.12, 'beta': 0.97, 'batch_size': 25, 'epochs': 50, 'degree': 2} → Accuracy: 0.8118
    #### ✅ Best Params: {'lr': 0.005, 'lambda_reg': 0.12, 'beta': 0.97, 'batch_size': 25, 'epochs': 50, 'degree': 2} → Accuracy: 0.8118
    #### ✅ Best Params: {'lr': 0.00025, 'lambda_reg': 0.08, 'beta': 0.925, 'batch_size': 50, 'epochs': 25, 'degree': 3} → Avg Accuracy: 0.8005
    #### ✅ Best Params: {'lr': 0.00025, 'lambda_reg': 0.12, 'beta': 0.95, 'batch_size': 25, 'epochs': 25, 'degree': 5} → Avg Accuracy: 0.8042
    #### ✅ Best Params: {'lr': 0.00025, 'lambda_reg': 0.1, 'beta': 0.95, 'batch_size': 25, 'epochs': 25, 'degree': 9} → Avg Accuracy: 0.8023
    print(f"\n🕒 Training Time: {end_time - start_time:.4f} seconds")

    # # Predict and evaluate
    # y_pred = predict_class(X_test, w)
    # accuracy = np.mean(y_pred == y_test)
    # print(f"\nTest Accuracy: {accuracy:.4f}")
    end_total_time = time.time()
    print(f"\n🕒 Total Time: {end_total_time - start_total_time:.4f} seconds")

    # # Confusion matrix
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.title("Confusion Matrix")
    # plt.show()