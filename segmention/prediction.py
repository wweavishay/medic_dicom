import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from model import CancerClassifier


def predict_and_color(image_path):
    img = cv2.imread(image_path)
    img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    img_tensor = transform(img_original)
    img_tensor = img_tensor.unsqueeze(0)

    model = CancerClassifier()
    model.load_state_dict(torch.load("saved_model.pth"))

    with torch.no_grad():
        model.eval()
        output = model(img_tensor)
        prediction = torch.sigmoid(output).item()
        if prediction > 0.5:
            print("The image contains cancer.")
            # Assuming you have a function to obtain the cancer mask from the model
            cancer_mask = get_cancer_mask(img_original)
            segmented_img = segment_and_color(img_original, cancer_mask)
            cv2.imshow("Segmented Image", segmented_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("The image does not contain cancer.")


def get_cancer_mask(img):
    # Placeholder for cancer segmentation
    # Assuming we have the cancer region as a binary mask where 1 represents cancerous areas
    cancer_mask = np.zeros_like(img[:, :, 0])  # Placeholder for demonstration
    return cancer_mask


def segment_and_color(img, cancer_mask):
    if np.any(cancer_mask):  # Check if there are any cancerous regions
        # Find contours in the cancer mask
        contours, _ = cv2.findContours(cancer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty mask to draw the contours on
        contour_mask = np.zeros_like(img)

        # Draw contours on the mask
        for contour in contours:
            # Draw contours filled with white color (255) on the mask
            cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Convert the mask to grayscale
        contour_mask_gray = cv2.cvtColor(contour_mask, cv2.COLOR_BGR2GRAY)

        # Apply a color map to the grayscale image
        colored_mask = cv2.applyColorMap(contour_mask_gray, cv2.COLORMAP_JET)

        # Add the colored mask to the original image
        segmented_img = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)

        return segmented_img

    else:
        print("No cancerous regions detected.")
        return img


if __name__ == "__main__":
    image_path = "17.jpg"
    predict_and_color(image_path)
