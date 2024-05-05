import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

# Check if GPU is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define directories
IMG_DIR = 'images'
TRAIN_DIR = os.path.join(IMG_DIR, 'train')
VAL_DIR = os.path.join(IMG_DIR, 'val')
TEST_DIR = os.path.join(IMG_DIR, 'test')

# Define classes dynamically from the test directory
CLASSES = sorted(os.listdir(TEST_DIR))

# Data transformations
transformations = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

# Custom dataset loader for the test directory
class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            class_dir = os.path.join(root, cls)
            for file_path in os.listdir(class_dir):
                if self.is_valid_file(file_path):
                    self.samples.append((os.path.join(class_dir, file_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def is_valid_file(self, file_path):
        return any(file_path.endswith(extension) for extension in ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'])

# Load datasets
train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_DIR, transform=transformations)
val_dataset = torchvision.datasets.ImageFolder(root=VAL_DIR, transform=transformations)
test_dataset = CustomImageFolder(root=TEST_DIR, transform=transformations)

# DataLoader
BATCH_SIZE = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*6*6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(CLASSES))
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Initialize model
model = CNN().to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training
NUM_EPOCHS = 20
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        yhat = model(x_batch)
        loss = loss_fn(yhat, y_batch)
        loss.backward()
        optimizer.step()

# Save trained model to a file
torch.onnx.export(model, torch.randn(1, 3, 100, 100).to(device), 'trained_model.onnx')


# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        x_test, y_test = data
        x_test, y_test = x_test.to(device), y_test.to(device)
        outputs = model(x_test)
        _, predicted = torch.max(outputs.data, 1)
        total += y_test.size(0)
        correct += (predicted == y_test).sum().item()

        # Get image names from the test loader dataset
        image_names = [os.path.basename(sample[0]) for sample in test_dataset.samples]

        # Print image names along with predictions
        for i in range(len(predicted)):
            print(f"Image: {image_names[i]}, Prediction: {CLASSES[predicted[i].item()]}")

# Calculate and print accuracy
accuracy = 100 * correct / total
print('Test Accuracy: {:.2f}%'.format(accuracy))
