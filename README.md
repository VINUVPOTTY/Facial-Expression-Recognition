# Facial Expression Recognition using Deep Learning

## Overview

This project implements a Facial Expression Recognition (FER) system using a Convolutional Neural Network (CNN) trained on the FER2013 dataset. The model is capable of classifying human facial expressions into seven emotion categories from grayscale facial images.

The project is developed using TensorFlow and Keras in Google Colab and includes data preprocessing, augmentation, model training, evaluation, and real-time prediction on new images.

## Features

* Facial expression classification using deep learning
* FER2013 dataset integration through Kaggle
* Image augmentation for improved generalization
* Class imbalance handling using class weights
* Model checkpointing and early stopping
* Learning rate scheduling
* Emotion prediction on custom images
* Face detection using OpenCV Haar Cascades

## Emotion Classes

The model predicts the following emotions:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

## Dataset

Dataset: FER2013

The dataset contains grayscale facial images of size 48×48 pixels categorized into seven emotion classes.

Source:
https://www.kaggle.com/datasets/msambare/fer2013

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab

## Project Workflow

### 1. Dataset Preparation

* Download FER2013 dataset using Kaggle API
* Extract dataset into training and testing directories

### 2. Data Preprocessing

* Rescale pixel values
* Apply data augmentation:

  * Rotation
  * Horizontal flipping

### 3. Class Balancing

Class weights are computed using Scikit-learn to reduce the impact of class imbalance during training.

### 4. Model Architecture

CNN Architecture:

* Input Layer (48 × 48 × 1)

* Conv2D (64 filters)

* Batch Normalization

* MaxPooling

* Dropout

* Conv2D (128 filters)

* Batch Normalization

* MaxPooling

* Dropout

* Conv2D (256 filters)

* Batch Normalization

* MaxPooling

* Dropout

* Flatten Layer

* Dense (256)

* Batch Normalization

* Dropout

* Dense (128)

* Batch Normalization

* Dropout

* Output Layer (7 classes with Softmax activation)

### 5. Training Configuration

* Optimizer: Adam
* Learning Rate: 0.001
* Loss Function: Categorical Crossentropy
* Metrics: Accuracy
* Epochs: Up to 70

Callbacks used:

* ModelCheckpoint
* ReduceLROnPlateau
* EarlyStopping

### 6. Evaluation

The best-performing model is saved automatically during training and evaluated on the test dataset.

## Files

| File                      | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| FER.ipynb                 | Complete project notebook                                |
| best_emotion_model.keras  | Best model saved during training                         |
| emotion_model_final.keras | Final trained model                                      |
| fer2013.zip               | FER2013 dataset                                          |
| kaggle.json               | Kaggle API credentials (should not be uploaded publicly) |

## Running the Project

### Install Dependencies

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn kaggle
```

### Download Dataset

Configure Kaggle API credentials and download the FER2013 dataset.

### Train the Model

Run all notebook cells sequentially to:

1. Download dataset
2. Preprocess images
3. Train CNN model
4. Save trained model
5. Evaluate performance

### Predict Emotion on New Images

Upload a face image and run the prediction section to classify the detected emotion.

## Future Improvements

* Transfer Learning using ResNet50, EfficientNet, or MobileNet
* Real-time webcam emotion detection
* Model deployment using Flask or Streamlit
* Ensemble learning for higher accuracy
* Hyperparameter optimization

## Author

Vinayak V Potty

## License

This project is intended for educational and research purposes.
