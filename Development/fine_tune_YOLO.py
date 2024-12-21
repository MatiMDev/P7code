import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
import matplotlib.pyplot as plt

# 1. Configuration
DATA_DIR = "yolo_data"
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data Preparation
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, img) for img in os.listdir(data_dir)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = plt.imread(img_path)
        label = 0 if "cat" in img_path else 1  # Example logic for binary classification
        
        if self.transform:
            image = self.transform(image)
        return image, label

# Apply transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images
])

# Load datasets
train_dataset = CustomDataset(os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = CustomDataset(os.path.join(DATA_DIR, "valid"), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Define the Model
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary classification
model = model.to(DEVICE)

# 4. Define Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Training Loop
def train_model():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

# 6. Validation Loop
def validate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Run Training and Validation
train_model()
validate_model()

# 7. Save the Model
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")
