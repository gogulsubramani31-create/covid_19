# COVID-19 Image Classification Using CNN

## 📌 Project Overview

This project develops a **Convolutional Neural Network (CNN)** based image classification system to identify COVID-19 cases from chest X-ray images.

The system classifies images into two categories:

* **COVID-19** – Patient affected by COVID-19
* **Normal** – Healthy person without COVID-19 symptoms

The project focuses on image preprocessing, CNN model development, model comparison, evaluation, and prediction on unseen images.

## 🎯 Objective

The main objective is to develop an AI-based diagnostic support system that can analyze chest X-ray images and identify patterns associated with COVID-19.

The system aims to:

* Support faster COVID-19 detection.
* Provide near real-time diagnostic insights.
* Assist healthcare professionals in patient triage.
* Reduce delays in identifying potential COVID-19 cases.
* Demonstrate the use of deep learning for medical image classification.

## 📊 Dataset

The dataset contains **251 images** belonging to two classes:

| Class    | Description               |
| -------- | ------------------------- |
| COVID-19 | COVID-19 affected patient |
| Normal   | Healthy person            |

### Image Details

* **Number of images:** 251
* **Image size:** 128 × 128 pixels
* **Channels:** 3 RGB channels
* **Input shape:** `(128, 128, 3)`

The dataset consists of:

* `CovidImages.npy` – Image data
* `CovidLabels.csv` – Corresponding labels

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* OpenCV
* TensorFlow
* Keras
* Scikit-learn
* Google Colab

## 📚 Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Input
)

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
```

## 🔄 Project Workflow

```text
Dataset
   ↓
Load Images and Labels
   ↓
Data Exploration
   ↓
Image Visualization
   ↓
Data Preprocessing
   ↓
Train / Validation / Test Split
   ↓
Image Normalization
   ↓
CNN Model 1
   ↓
CNN Model 2
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Final Model Selection
   ↓
Prediction on Unseen Images
```

## ⚙️ Data Preprocessing

The images are prepared before training the CNN model.

The dataset images are normalized so that pixel values are converted to a range of **0 to 1**. The notebook confirms that the normalized training, validation, and test sets have minimum and maximum values of `0.0` and `1.0`.

A random seed of **42** is also used to improve consistency during experimentation.

## 🧠 CNN Models

Two CNN architectures were developed and compared.

### Model 1 – 3-Layer CNN

Architecture:

```text
Input: 128 × 128 × 3
        ↓
Conv2D - 32 filters
        ↓
MaxPooling
        ↓
Conv2D - 64 filters
        ↓
MaxPooling
        ↓
Conv2D - 128 filters
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense - 128
        ↓
Sigmoid Output
```

The model contains approximately **4.29 million trainable parameters**.

### Model 2 – 2-Layer CNN

To reduce model complexity and improve generalization, the second model uses two convolutional and pooling combinations.

```text
Input: 128 × 128 × 3
        ↓
Conv2D - 32 filters
        ↓
MaxPooling
        ↓
Conv2D - 64 filters
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense - 128
        ↓
Dense - 64
        ↓
Sigmoid Output
```

The model uses:

* Adam optimizer
* Learning rate: `0.0001`
* Binary cross-entropy loss
* Accuracy as the evaluation metric
* ReLU activation in hidden layers
* Sigmoid activation in the output layer

The final model contains **8,416,449 trainable parameters**.

## 📈 Model Performance

| Model               | Train Accuracy | Validation Accuracy | COVID Recall | Validation COVID Recall |
| ------------------- | -------------: | ------------------: | -----------: | ----------------------: |
| CNN – 3 Conv Layers |           100% |              97.37% |         100% |                    100% |
| CNN – 2 Conv Layers |            96% |              97.37% |       90.91% |                  94.12% |

The 3-layer CNN achieved perfect training performance but showed signs of overfitting. The 2-layer CNN achieved the same validation accuracy with lower training performance, indicating better generalization.

## 🏆 Final Model

The **2-Layer CNN** was selected as the final model because it provided strong validation performance while reducing the risk of overfitting compared with the deeper CNN.

### Final Test Results

* **Test Accuracy:** 97.37%
* **COVID Recall:** 94.12%

These results show that the selected CNN performs well on unseen test images.

## 🔍 Evaluation Metrics

Two important metrics were considered:

### Accuracy

Measures the percentage of correctly classified images across both COVID-19 and Normal classes.

### COVID Recall

Measures how effectively the model identifies actual COVID-positive cases.

COVID recall is particularly important for this classification task because missing a COVID-positive case represents a false negative.

## 🖼️ Prediction on Unseen Images

The trained model is also used to classify an unseen image.

OpenCV is used to:

1. Load the input image.
2. Resize the image.
3. Display the image.
4. Prepare it for prediction using the trained CNN.

The notebook demonstrates prediction using an external COVID-19 image.

## 📁 Project Structure

```text
COVID-19-Image-Classification/
│
├── Covid_19_Image_Classification.ipynb
├── CovidImages.npy
├── CovidLabels.csv
├── covid19.webp
└── README.md
```

> **Note:** The dataset files are not necessarily included in the GitHub repository because of their size and data-related considerations.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd COVID-19-Image-Classification
```

### 2. Open the notebook

The project is designed to run in **Google Colab**.

Upload/open:

```text
Covid_19_Image_Classification.ipynb
```

### 3. Mount Google Drive

The notebook loads the dataset from Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Place the dataset files

Ensure the following files are available:

```text
CovidImages.npy
CovidLabels.csv
```

### 5. Run the notebook

Execute the notebook cells sequentially to:

* Load the dataset
* Explore the images
* Preprocess the data
* Train CNN models
* Compare model performance
* Evaluate the final model
* Predict unseen images

## ⚠️ Limitations

* The dataset contains only **251 images**, which is relatively small for training a deep learning model.
* The 3-layer CNN showed signs of overfitting.
* The notebook does not save the trained model, so retraining can produce slightly different results.
* CNN results may vary because of weight initialization, data shuffling, and GPU-related non-determinism.
* This project should be considered an **AI research/educational project**, not a replacement for professional medical diagnosis.

## 🚀 Future Improvements

* Increase the size and diversity of the dataset.
* Apply data augmentation techniques.
* Use transfer learning models such as ResNet, VGG, or EfficientNet.
* Add regularization techniques such as Dropout.
* Use cross-validation for more reliable evaluation.
* Save and reuse the trained model.
* Develop a web-based interface for image prediction.
* Evaluate the model using additional medical classification metrics.



## 📌 Conclusion

This project demonstrates how CNN-based deep learning can be applied to classify chest X-ray images into COVID-19 and Normal categories. Two CNN architectures were compared, and the **2-layer CNN** was selected as the final model based on its generalization performance.

The final model achieved **97.37% test accuracy** and **94.12% COVID recall** on the available test dataset.
