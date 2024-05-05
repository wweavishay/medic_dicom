import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Function to preprocess a single image
def preprocess_image_single(image_path, target_size=(150, 150)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    img = cv2.resize(img, target_size)
    img = img.flatten()
    return img

# Function to preprocess images in a folder
def preprocess_images_folder(folder_path, target_size=(150, 150)):
    images = []
    image_names = os.listdir(folder_path)
    for image_name in image_names:
        image_path = os.path.join(folder_path, image_name)
        try:
            img = preprocess_image_single(image_path, target_size)
            images.append((image_name, img))  # Store image name along with image data
        except Exception as e:
            print(f"Error processing image: {image_name}. {e}")
    return images

# Function to predict either a single image or images in a folder
def predict_images(input_path, model):
    if os.path.isdir(input_path):
        images = preprocess_images_folder(input_path)
    else:
        images = [(os.path.basename(input_path), preprocess_image_single(input_path))]  # Use basename for single image
    predictions = []
    for image_name, img in images:
        prediction = model.predict([img])[0]
        predictions.append((image_name, prediction))
    return predictions

def main():
    # Load the trained classifier
    try:
        model = joblib.load('random_forest_model.pkl')  # Load the model from pickle file
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Ask the user for input type
    while True:
        user_input = input("Enter 'folder' command if you want to input a folder containing multiple images, Or \n'file' command if you want to input a single image file: ").lower()
        if user_input == 'folder' or user_input == 'file':
            break
        else:
            print("Invalid input. Please enter 'folder' or 'file'.")

    # Perform prediction based on user input
    if user_input == 'folder':
        folder_path = input("Enter the path to the folder containing images: ")
        predictions = predict_images(folder_path, model)
        print("Predictions for images in folder:")
        for image_name, prediction in predictions:
            print(f"Image: {image_name}, Prediction: {prediction}")
    else:
        image_path = input("Enter the path to the image file: ")
        prediction = predict_images(image_path, model)
        print("Prediction for single image:")
        print(prediction)

if __name__ == "__main__":
    main()
