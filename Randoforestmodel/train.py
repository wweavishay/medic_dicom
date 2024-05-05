import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

def preprocess_image(img_path, target_size=(150, 150)):
    print("Processing image:", img_path)  # Print image path for debugging
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, target_size)
        img = img.flatten()
        return img
    except Exception as e:
        print(f"Error processing image: {img_path}. {e}")
        return None

# Load the data from folders recursively
def load_data(folder_path):
    data = []
    labels = []
    for root, dirs, files in os.walk(folder_path):
        for folder_name in dirs:
            sub_folder_path = os.path.join(root, folder_name)
            print(f"Loading images from folder: {sub_folder_path}")
            images = os.listdir(sub_folder_path)
            for image_name in images:
                if image_name.lower().endswith('.png') or image_name.lower().endswith('.jpeg') or image_name.lower().endswith('.jpg'):
                    # Skip files with additional characters or prefixes
                    if not image_name.startswith('._'):
                        try:
                            img_path = os.path.join(sub_folder_path, image_name)
                            img = preprocess_image(img_path)
                            if img is not None:  # Ensure image was loaded successfully
                                data.append(img)
                                labels.append(folder_name)
                        except Exception as e:
                            print(f"Error processing image: {image_name}. {e}")
    return np.array(data), np.array(labels)


def main():
    folder_path = 'imageskidney'

    # Load the data
    try:
        data, labels = load_data(folder_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    # Define the pipeline with regularization
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features by removing the mean and scaling to unit variance
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))  # RandomForestClassifier with 100 estimators
    ])

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform stratified k-fold cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf)

    print("Cross-Validation Accuracy Scores:", cv_scores)
    print("Mean CV Accuracy:", np.mean(cv_scores))

    # Train the final model
    pipeline.fit(X_train, y_train)

    # Evaluate the classifier on the test set
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy}")

    # Plotting
    plt.bar(range(len(cv_scores)), cv_scores)
    plt.title('Cross-Validation Accuracy Scores')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.show()

    # Save the trained classifier
    joblib.dump(pipeline, 'random_forest_model.pkl')
    print("Model saved as 'random_forest_model.pkl'.")

    # Save parameters and accuracies to log file
    with open('model_log.txt', 'w') as f:
        f.write(f"Cross-Validation Accuracy Scores: {cv_scores}\n")
        f.write(f"Mean CV Accuracy: {np.mean(cv_scores)}\n")
        f.write(f"Test Accuracy: {test_accuracy}\n")

if __name__ == "__main__":
    main()
