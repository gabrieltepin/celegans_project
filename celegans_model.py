import pickle
import numpy as np
import os
from PIL import Image
import pandas as pd

###############################
#   Auxiliary Functions       #
###############################

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w):
    return sigmoid(X @ w)

def predict_class(X, w, threshold=0.5):
    probs = predict_proba(X, w)
    return (probs >= threshold).astype(int)

def PHI(X, m):
    """
    Same polynomial expansion used in training.
    """
    X_no_bias = X[:, :-1]   # remove last column (bias)
    bias = X[:, -1:]        # save bias to reattach

    features = [X_no_bias ** d for d in range(1, m + 1)]
    X_poly = np.concatenate(features, axis=1)
    X_poly = np.hstack((X_poly, bias))  # reattach bias column
    return X_poly

def preprocess_new_images(base_path, img_size=(28, 28)):
    """
    Loads images from `base_path` and reshapes them into the same design 
    as in training (flatten + add bias).
    """
    image_paths = []
    for filename in sorted(os.listdir(base_path)):
        if filename.endswith(".png"):
            image_paths.append(os.path.join(base_path, filename))

    # If no .png files found, handle gracefully
    if len(image_paths) == 0:
        print(f"No PNG images found in {base_path}!")
        return np.array([]), []

    X = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('L')
        img = img.resize(img_size)
        img_array = np.array(img).astype(np.float32) / 255.0
        X.append(img_array.flatten())

    X = np.array(X)
    # Add bias as last column
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    return X, image_paths

def apply_pca(X, pcs, mean_pca):
    """
    Applies the PCA transformation with the principal components (pcs)
    and the mean used during training.
    """
    bias = X[:, -1].reshape(-1, 1)
    X_centered = X[:, :-1] - mean_pca
    X_pca = np.dot(X_centered, pcs)
    return np.hstack((X_pca, bias))


############################
#   Main Celegans Script  #
############################

if __name__ == "__main__":
    # --- Load the trained model parameters ----
    with open("celegans_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Extract needed items
    w = model['w']
    pcs = model['pcs']
    mean_pca = model['mean_pca']
    mean_train = model['mean_train']
    std_train = model['std_train']
    phi_degree = model['phi_degree']

    # base_path = "Celegans_ModelGen/0"  # folder with new .png images
    # base_path = "test_celegans"  # folder with new .png images
    base_path = input("Enter the path to the image folder: ").strip()
    while not os.path.isdir(base_path):
        print("That folder doesn't exist. Try again.")
        base_path = input("Enter the path to the image folder: ").strip()
    X_raw, img_paths = preprocess_new_images(base_path)
    if X_raw.size == 0:
        print("No images to classify. Exiting.")
        exit(0)

    # 3) Apply the same PCA used in training
    X_pca = apply_pca(X_raw, pcs, mean_pca)

    # 4) Apply the same PHI expansion
    X_phi = PHI(X_pca, phi_degree)

    # 5) Normalize using the training set's mean/std
    X_features = X_phi[:, :-1]
    X_bias = X_phi[:, -1:]
    X_features = (X_features - mean_train) / std_train
    X_final = np.hstack((X_features, X_bias))

    # 6) Predict classes
    y_pred = predict_class(X_final, w)

    # 7) Prepare a DataFrame with columns [image_name, label]
    data = {
        "image_name": [os.path.basename(p) for p in img_paths],
        "label": y_pred
    }
    df = pd.DataFrame(data)

    # 8) Count the labels 0/1 and append summary rows to the bottom
    num_label_0 = (y_pred == 0).sum()
    num_label_1 = (y_pred == 1).sum()

    # Append as extra rows
    df.loc[len(df)] = ["TOTAL label 0", num_label_0]
    df.loc[len(df)] = ["TOTAL label 1", num_label_1]

    # 9) Save to Excel file
    output_file = "celegans_result.xlsx"
    df.to_excel(output_file, index=False)

    print(f"\nClassification complete. Results saved to {output_file}")
