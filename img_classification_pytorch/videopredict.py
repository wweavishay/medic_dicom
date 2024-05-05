import onnx
import onnxruntime
import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image
import os

IMG_DIR = 'images'
TEST_DIR = os.path.join(IMG_DIR, 'test')
CLASSES = sorted(os.listdir(TEST_DIR))

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

onnx_model = onnx.load('trained_model.onnx')
ort_session = onnxruntime.InferenceSession('trained_model.onnx')

def predict_image(img_tensor, threshold=50):
    try:
        ort_inputs = {ort_session.get_inputs()[0].name: img_tensor.numpy().astype(np.float32)}
        ort_outs = ort_session.run(None, ort_inputs)
        output = ort_outs[0]
        _, predicted = torch.max(torch.from_numpy(output), 1)
        confidence = torch.softmax(torch.from_numpy(output), dim=1)[0][predicted.item()] * 100
        prediction = CLASSES[predicted.item()]
        return ("Unknown", confidence.item()) if confidence < threshold else (prediction, confidence.item())
    except Exception as e:
        print("Error occurred while processing the image.")
        return "Error", 0.0

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    return transform(img_pil).unsqueeze(0)

def draw_prediction(frame, prediction, confidence):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (30, 380)
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2
    cv2.putText(frame, prediction + "- " + str(int(confidence)) + "%", org, font, font_scale, color, thickness,
                cv2.LINE_AA)
    return frame

def main():
    video_path = 'output_video.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
    else:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fps = fps // 2

        out = cv2.VideoWriter('PREDICT_TYPE_output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_tensor = process_frame(frame)
            prediction, confidence = predict_image(img_tensor)
            print(f"Predicted category: {prediction}, Confidence: {confidence:.2f}%")

            frame = draw_prediction(frame, prediction, confidence)
            out.write(frame)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
