import onnx
import onnxruntime
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import os

IMG_DIR = 'images'
TEST_DIR = os.path.join(IMG_DIR, 'test')
# Define classes dynamically from the test directory
CLASSES = sorted(os.listdir(TEST_DIR))

# Define transformations
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

# Load the ONNX model
onnx_model = onnx.load('trained_model.onnx')

# Create an inference session with the ONNX model
ort_session = onnxruntime.InferenceSession('trained_model.onnx')

# Function to predict the image category
def predict_image(image_path, threshold=80):
    try:
        # Open and preprocess the image
        img = Image.open(image_path)
        img_tensor = transform(img).unsqueeze(0)

        # Perform prediction
        ort_inputs = {ort_session.get_inputs()[0].name: img_tensor.numpy().astype(np.float32)}
        ort_outs = ort_session.run(None, ort_inputs)
        output = ort_outs[0]
        _, predicted = torch.max(torch.from_numpy(output), 1)
        confidence = torch.softmax(torch.from_numpy(output), dim=1)[0][predicted.item()] * 100
        prediction = CLASSES[predicted.item()]

        if confidence < threshold:
            print(f"Confidence level: {confidence:.2f}% (Below threshold)")
            print("I don't know how to classify this image.")
        else:
            print(f"Confidence level: {confidence:.2f}%")
            print(f"Predicted category for {image_path}: {prediction}")
    except Exception as e:
        print("Error occurred while processing the image.")

# Example usage
image_path = '1.jpg'  # Replace with the path to your new image
predict_image(image_path)
