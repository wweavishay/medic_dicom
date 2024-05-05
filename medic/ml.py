import numpy as np
import pandas as pd
import psycopg2
import pydicom
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from io import BytesIO
from PIL import Image
import io

# Function to retrieve DICOM images and labels from the database
def get_dicom_data(conn):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT dm.image_width, dm.image_height, df.dicom_data, dm.disease_type
        FROM dicom_metadata dm
        JOIN dicom_files df ON dm.id = df.metadata_id
    """)
    rows = cursor.fetchall()
    cursor.close()
    X = []
    y = []
    for row in rows:
        # Process DICOM image
        dicom_data = pydicom.dcmread(BytesIO(row[2]))
        image = np.array(dicom_data.pixel_array) / 255.0  # Normalize pixel values
        X.append(image)
        # Process disease type
        y.append(row[3])
    return np.array(X), np.array(y)

# Connect to PostgreSQL database
conn = psycopg2.connect(database="your_db_name", user="your_username", password="your_password", host="localhost", port="5432")

# Retrieve DICOM images and labels from the database
X, y = get_dicom_data(conn)

# Close database connection
conn.close()

# Encode disease types
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape data for CNN input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Make predictions
predictions = np.argmax(model.predict(X_test), axis=-1)
predicted_classes = label_encoder.inverse_transform(predictions)

# Function to predict disease type from a new DICOM image
def predict_disease_type(image_path):
    # Read DICOM image
    dicom_data = pydicom.dcmread(image_path)
    image = np.array(dicom_data.pixel_array) / 255.0  # Normalize pixel values
    image = image.reshape(1, image.shape[0], image.shape[1], 1)  # Reshape for CNN input

    # Make prediction
    prediction = model.predict(image)
    predicted_class = label_encoder.inverse_transform(np.argmax(prediction, axis=-1))
    return predicted_class[0]

# Example usage
new_image_path = "path/to/your/new/dicom/image.dcm"
predicted_disease_type = predict_disease_type(new_image_path)
print("Predicted Disease Type:", predicted_disease_type)
