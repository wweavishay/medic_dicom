# Image Classification Pipeline with Random Forest

## Introduction
This document outlines a Python script for building an image classification pipeline using Random Forest algorithm. The pipeline includes image preprocessing, model training, cross-validation, evaluation, and inference functionalities.

## Libraries Used
- OpenCV (`cv2`): For image loading and preprocessing.
- NumPy (`numpy`): For numerical operations and array manipulation.
- Matplotlib (`matplotlib.pyplot`): For data visualization.
- Scikit-learn (`sklearn`): For machine learning tasks such as classification, cross-validation, and preprocessing.
- Joblib (`joblib`): For saving and loading trained models.

## Image Preprocessing
- `preprocess_image`: Function to preprocess a single image, including reading, resizing, and flattening.
- `load_data`: Function to load image data from a specified folder recursively, preprocess them, and store along with labels.

## Model Training and Evaluation
- `main`: Main function orchestrating the training and evaluation process.
  - Load image data using `load_data`.
  - Split data into training and testing sets.
  - Create a machine learning pipeline with scaling and Random Forest classifier.
  - Perform stratified k-fold cross-validation for model evaluation.
  - Train the final model and evaluate it on the test set.
  - Save the trained model and evaluation results.

## Image Prediction
- `preprocess_image_single` and `preprocess_images_folder`: Functions to preprocess a single image or multiple images in a folder.
- `predict_images`: Function to predict class labels of images, handling both single image and multiple images scenarios.

## Model Inference
- Load the trained model.
- Prompt the user to input whether they want to predict images from a folder or a single image file.
- Call the appropriate prediction function based on user input and display the predictions.



# Predict.py Script Usage Guide

The `predict.py` script is designed to predict the class labels of images using a pre-trained Random Forest Classifier. This guide provides step-by-step instructions on how to use the script effectively.

## Running the Script
1. **Prepare the Model**:
   Ensure that you have a trained Random Forest Classifier saved as `random_forest_model.pkl` in the same directory as the `predict.py` script.

2. **Execute the Script**:
   Open your terminal or command prompt, navigate to the directory containing the `predict.py` script, and run the script by typing: