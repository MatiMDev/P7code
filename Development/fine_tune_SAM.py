import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
import os
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

# Define Dataset
class RopeDamageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, torch.tensor(np.array(mask) / 255.0, dtype=torch.float32)

# Directories
image_dir = "dataset\small_datasets\Chafing\Chafing_High"
mask_dir = "dataset\small_datasets\Chafing\Chafing_High\smaller_dataset_labels_coco.json"

# Transforms
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])

# Dataset and DataLoader
dataset = RopeDamageDataset(image_dir, mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint="/path/to/sam_vit_h.pth")
sam.to(device)

# Modify the SAM model if needed
sam.prompt_encoder.requires_grad_(False)  # Freeze prompt encoder
sam.image_encoder.requires_grad_(True)   # Fine-tune image encoder
sam.mask_decoder.requires_grad_(True)    # Fine-tune mask decoder

# Loss Function and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, sam.parameters()), lr=1e-4)

# Training Loop
epochs = 10
sam.train()
for epoch in range(epochs):
    epoch_loss = 0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        embeddings = sam.image_encoder(images)
        outputs = sam.mask_decoder(embeddings)
        
        # Calculate loss
        loss = criterion(outputs, masks.unsqueeze(1))  # Unsqueeze for channel dimension
        epoch_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Save the Fine-Tuned Model
torch.save(sam.state_dict(), "fine_tuned_sam.pth")

print("Training complete!")
