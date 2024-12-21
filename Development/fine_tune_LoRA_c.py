import os
import sys
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from segment_anything import (
    SamPredictor,
    sam_model_registry,
)

class CustomSAMDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        """
        Args:
        - json_path: Path to COCO annotations JSON file.
        - image_dir: Directory containing the images.
        - transform: Optional transforms to be applied on a sample.
        """
        self.coco = COCO(json_path)
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create binary mask from annotations
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

        for ann in annotations:
            segmentation = ann['segmentation']
            if isinstance(segmentation, list):  # Polygon format
                for polygon in segmentation:
                    poly = np.array(polygon).reshape((-1, 2))
                    cv2.fillPoly(mask, [poly.astype(int)], color=1)

        # Prepare point inputs (center of the mask)
        h, w = mask.shape
        mask_indices = np.column_stack(np.where(mask > 0))
        if len(mask_indices) > 0:
            points = np.mean(mask_indices, axis=0).astype(int)
        else:
            points = np.array([h // 2, w // 2])  # Default point if no mask

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return {
            'image': torch.from_numpy(image).permute(2, 0, 1).float(),
            'mask': torch.from_numpy(mask).float(),
            'point_coords': torch.from_numpy(points).float(),
            'point_labels': torch.tensor([1])  # Positive point
        }

class SAMLossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_masks, true_masks):
        """
        Compute loss between predicted and ground truth masks.

        Args:
        - pred_masks: Predicted masks from SAM
        - true_masks: Ground truth masks
        """
        return self.bce_loss(pred_masks, true_masks)

def setup_sam_model(model_type='vit_h', checkpoint_path=None):
    """
    Initialize SAM model.

    Args:
    - model_type: Type of SAM model (vit_h, vit_l, vit_b)
    - checkpoint_path: Path to pretrained checkpoint
    """
    sam = sam_model_registry[model_type]()

    if checkpoint_path:
        sam.load_state_dict(torch.load(checkpoint_path))

    return sam

def train_sam(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10):
    """
    Train SAM model.

    Args:
    - model: SAM model
    - train_loader: Training data loader
    - val_loader: Validation data loader
    - optimizer: Optimizer
    - loss_fn: Loss function
    - device: Training device
    - epochs: Number of training epochs
    """
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_train_loss = 0.0

        for batch in train_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            point_coords = batch['point_coords'].to(device)  # Shape: [batch_size, 2]
            point_labels = batch['point_labels'].to(device)  # Shape: [batch_size]

            # Prepare points as a tuple of (coords, labels)
            points = (point_coords.unsqueeze(1), point_labels.unsqueeze(1))

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points, boxes=None, masks=None
            )

            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=model.image_encoder(images),
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings
            )

            # Compute loss
            loss = loss_fn(low_res_masks, masks)
            total_train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                point_coords = batch['point_coords'].to(device)
                point_labels = batch['point_labels'].to(device)

                # Prepare points as a tuple of (coords, labels)
                points = (point_coords.unsqueeze(1), point_labels.unsqueeze(1))

                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=points, boxes=None, masks=None
                )

                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=model.image_encoder(images),
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings
                )

                val_loss = loss_fn(low_res_masks, masks)
                total_val_loss += val_loss.item()

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {total_train_loss/len(train_loader)}")
        print(f"Validation Loss: {total_val_loss/len(val_loader)}")

    return model

def evaluate_sam(model, test_loader, device, output_dir):
    """
    Evaluate SAM model and visualize results.

    Args:
    - model: Trained SAM model
    - test_loader: Test data loader
    - device: Evaluation device
    - output_dir: Directory to save evaluation results
    """
    model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            true_masks = batch['mask'].to(device)
            point_coords = batch['point_coords'].to(device)
            point_labels = batch['point_labels'].to(device)

            # Inference
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=point_coords, boxes=None, masks=None
            )

            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=model.image_encoder(images),
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings
            )

            # Convert predictions to binary masks
            pred_masks = (low_res_masks > 0.5).float()

            # Visualization
            for i in range(images.shape[0]):
                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.title('Original Image')
                plt.imshow(images[i].cpu().permute(1, 2, 0))

                plt.subplot(1, 3, 2)
                plt.title('Predicted Mask')
                plt.imshow(pred_masks[i].cpu(), cmap='gray')

                plt.subplot(1, 3, 3)
                plt.title('Ground Truth Mask')
                plt.imshow(true_masks[i].cpu(), cmap='gray')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'result_{idx}_{i}.png'))
                plt.close()

def main():
    # Logging setup
    logging.basicConfig(level=logging.INFO)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    MODEL_TYPE = 'vit_h'  # Can be 'vit_h', 'vit_l', or 'vit_b'

    # Paths
    TRAIN_JSON = "CutStrands.v1i.coco-segmentation/train/_annotations.coco.json"
    TEST_JSON = "CutStrands.v1i.coco-segmentation/test/_annotations.coco.json"
    VALID_JSON = "CutStrands.v1i.coco-segmentation/valid/_annotations.coco.json"
    IMAGE_DIR_TRAIN = "CutStrands.v1i.coco-segmentation/train"
    IMAGE_DIR_TEST = "CutStrands.v1i.coco-segmentation/test"
    IMAGE_DIR_VALID = "CutStrands.v1i.coco-segmentation/valid"

    # Output directory
    OUTPUT_DIR = os.path.join('outputs', f'sam_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Datasets and data loaders
    train_dataset = CustomSAMDataset(TRAIN_JSON, IMAGE_DIR_TRAIN)
    val_dataset = CustomSAMDataset(VALID_JSON, IMAGE_DIR_VALID)
    test_dataset = CustomSAMDataset(TEST_JSON, IMAGE_DIR_TEST)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Model setup
    checkpoint_path = 'sam_vit_h_4b8939.pth'  # Optional
    sam_model = setup_sam_model(model_type=MODEL_TYPE, checkpoint_path=checkpoint_path)

    # Loss and optimizer
    loss_function = SAMLossFunction()
    optimizer = optim.Adam(sam_model.parameters(), lr=LEARNING_RATE)

    # Train model
    trained_model = train_sam(
        sam_model,
        train_loader,
        val_loader,
        optimizer,
        loss_function,
        device,
        epochs=EPOCHS
    )

    # Save model
    torch.save(trained_model.state_dict(), os.path.join(OUTPUT_DIR, 'sam_model.pth'))

    # Evaluate and visualize
    evaluate_sam(trained_model, test_loader, device, OUTPUT_DIR)

if __name__ == "__main__":
    main()
