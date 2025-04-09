import os
from itertools import product
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import time
import pickle
import seaborn as sns

np.random.seed(11)

def get_celegans_image_paths(base_path: str, classes: list[str]) -> dict:
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

def train_test_split_manual(X, y, test_size=0.1, random_state=None):
    n_samples = len(X)
    if isinstance(test_size, float):
        test_size = int(n_samples * test_size)

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_test = X[test_indices]
    y_test = y[test_indices]
    X_train = X[train_indices]
    y_train = y[train_indices]

    return X_train, X_test, y_train, y_test

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
    X_pca = np.hstack((X_reduced, bias)) 

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
    X_ = X[:, :-1]          # remove last column (bias)
    bias = X[:, -1:]        # save bias to reattach

    features = [X_**d for d in range(1, m + 1)]
    X = np.concatenate(features, axis=1)
    X = np.hstack((X, bias))  # reattach bias column
    return X

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

    print(f"\nBest Params: {best_params} → Accuracy: {best_accuracy:.4f}")
    return best_params, best_accuracy

def compute_confusion_matrix(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Mismatched lengths!"
    
    TN = FP = FN = TP = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            TN += 1
        elif yt == 0 and yp == 1:
            FP += 1
        elif yt == 1 and yp == 0:
            FN += 1
        elif yt == 1 and yp == 1:
            TP += 1

    return np.array([[TN, FP],
                     [FN, TP]])


if __name__ == "__main__": 
    start_total_time = time.time()
    
    # ---------- Preprocessing ----------
    start_prep_time = time.time()
    base_path = "Celegans_ModelGen"
    class_labels = ['0', '1']
    image_paths = get_celegans_image_paths(base_path, class_labels)

    X, y = load_celegans_design_matrix(image_paths, img_size=(28, 28))
    # X, y = load_celegans_design_matrix(image_paths, img_size=(28, 28), max_per_class=200)

    X, pcs, mean_pca = pca(X, n_components=500)

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.1, random_state=11)
    end_prep_time = time.time()
    print(f"\nPreprocessing Time: {end_prep_time - start_prep_time:.4f} seconds")

    # ---------- Training ----------
    start_exec_time = time.time()
    param_grid = {'lr': [0.0025],'lambda_reg': [0.1],'beta': [0.97],'batch_size': [50],'epochs': [50],'degree': [5]}
    best_params, best_acc = grid_search_logistic_regression(X, y, param_grid)
    end_exec_time = time.time()
    print(f"\nTraining Time: {end_exec_time - start_exec_time:.4f} seconds")

    # ---------- Evaluate ----------
    X_train_phi = PHI(X_train, best_params['degree'])
    X_test_phi = PHI(X_test, best_params['degree'])
    X_train_phi, mean_train, std_train = normalize_features(X_train_phi)
    X_test_features = X_test_phi[:, :-1]
    X_test_bias = X_test_phi[:, -1:]
    X_test_features = (X_test_features - mean_train) / std_train
    X_test_phi = np.hstack((X_test_features, X_test_bias))
    w = train_logistic_regression_sgd(
        X_train_phi,
        y_train,
        lr=best_params['lr'],
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        beta=best_params['beta'],
        lambda_reg=best_params['lambda_reg'],
        verbose=True
    )
    y_pred = predict_class(X_test_phi, w)
    accuracy = np.mean(y_pred == y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    end_total_time = time.time()
    print(f"\nTotal Time: {end_total_time - start_total_time:.4f} seconds")

    # ---------- Confusion Martix ----------
    cm = compute_confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # ---------- Saving the model ----------
    model_dict = {
        'w': w,
        'pcs': pcs,
        'mean_pca': mean_pca,
        'mean_train': mean_train,
        'std_train': std_train,
        'phi_degree': best_params['degree']
    }

    with open("celegans_model.pkl", "wb") as f:
        pickle.dump(model_dict, f)
