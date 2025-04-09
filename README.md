# Image Classification: MNIST & C. Elegans Logistic Models

This repository contains two image classification models:

- `mnist_model.py` — Classifies handwritten digits (0–9) using a trained softmax regression model with HOG features.
- `celegans_model.py` — Classifies grayscale microscopy images as either **worm-present (1)** or **no-worm (0)** using logistic regression with PCA + polynomial features.


## Run
#### MNIST
```
python mnist_model.py
```
You’ll be prompted:
```
Enter the path to the image folder:
```
So enter the path to the MNIST _.tif_ images and it will generate a mnist_results.xlsx excel file classifying the images in the specfied folder.

#### Celegans
```
python celegans_model.py
```
You’ll be prompted:
```
Enter the path to the image folder:
```
So enter the path to the Celegans _.png_ images and it will generate a celegans_results.xlsx excel file classifying the images in the specfied folder.
