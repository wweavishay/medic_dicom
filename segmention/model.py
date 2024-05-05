import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# Step 1: Data Preparation
class CustomDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images, self.labels = self.load_data()

    def load_data(self):
        images = []
        labels = []
        for label, category in enumerate(os.listdir(self.folder)):
            category_folder = os.path.join(self.folder, category)
            for filename in os.listdir(category_folder):
                img = cv2.imread(os.path.join(category_folder, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.transform:
                    img = self.transform(img)
                images.append(img)
                labels.append(label)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# Step 2: Model Definition
class CancerClassifier(nn.Module):
    def __init__(self):
        super(CancerClassifier, self).__init__()
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()  # Adding sigmoid activation function
        )

    def forward(self, x):
        x = self.features(x)
        return x


# Step 3: Model Training
def train_model(model, criterion, optimizer, train_loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Loss: {epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "saved_model.pth")


# Step 4: Semantic Segmentation (Using pre-trained model)
def semantic_segmentation(image_tensor):
    # Placeholder random segmentation mask for demonstration
    return torch.rand_like(image_tensor[:, 0, :, :])


# Step 5: Post-Processing
def post_process_segmentation(segmentation_mask):
    # Placeholder post-processing for demonstration
    return segmentation_mask


# Step 6: Visualization
def visualize_segmentation(original_img, segmentation_mask):
    # Ensure segmentation mask is in the same shape as the original image
    segmentation_mask = segmentation_mask.squeeze().numpy()  # Squeeze to remove batch dimension and convert to numpy array

    # Overlay the segmented regions on the original image for visualization
    segmented_img = original_img.copy()
    segmented_img[segmentation_mask > 0.5] = [255, 0, 0]  # Highlight cancerous regions in red
    return segmented_img


def main():
    # Define data transformations for augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(50),
        transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.1, hue=0.1)
    ])

    # Load data
    train_dataset = CustomDataset("dataset/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create model
    model = CancerClassifier()

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Using Binary Cross Entropy Loss since we're adding sigmoid activation
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    train_model(model, criterion, optimizer, train_loader)

    # Example of semantic segmentation and visualization
    image_tensor, _ = train_dataset[0]  # Sample image tensor
    segmentation_mask = semantic_segmentation(image_tensor.unsqueeze(0))
    refined_mask = post_process_segmentation(segmentation_mask)
    visualization = visualize_segmentation(image_tensor.permute(1, 2, 0).numpy(), refined_mask)
    plt.imshow(visualization)
    plt.show()


if __name__ == "__main__":
    main()
