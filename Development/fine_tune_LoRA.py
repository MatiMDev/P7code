import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np

# Paths
COCO_DATASET_PATH = "dataset/small_datasets/Chafing/Chafing_High/smaller_dataset_labels_coco.json"
OUTPUT_DIR = "trained_models"


class CocoDataset(Dataset):
    """Dataset class for COCO-style annotations."""

    def __init__(self, coco_file, image_dir):
        with open(coco_file, "r") as f:
            self.coco_data = json.load(f)
        self.image_dir = image_dir
        self.annotations = self.coco_data["annotations"]
        self.images = {img["id"]: img for img in self.coco_data["images"]}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_info = self.images[annotation["image_id"]]
        image_path = os.path.join(self.image_dir, image_info["file_name"])

        # Read and normalize the image
        image = read_image(image_path).float() / 255.0  # Normalize to [0, 1] range

        # Create segmentation mask
        mask = self.create_mask(annotation, image.shape[1:])

        return image, mask

    def create_mask(self, annotation, image_shape):
        """Creates a binary mask from the COCO segmentation."""
        mask = np.zeros(image_shape, dtype=np.float32)
        segmentation = annotation["segmentation"]
        for polygon in segmentation:
            points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [points], color=1.0)
        return torch.tensor(mask, dtype=torch.float32)  # Ensure mask is float for loss computation


def train_lora_sam(coco_file, image_dir, output_dir, model_checkpoint, epochs=10, batch_size=4, lr=1e-4):
    """Train LoRA-SAM with the given COCO dataset."""
    # Load the SAM model and apply LoRA
    sam = sam_model_registry["vit_b"](checkpoint=model_checkpoint)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")

    # Wrap SAM predictor with LoRA
    predictor = SamPredictor(sam)

    # Prepare dataset and dataloader
    dataset = CocoDataset(coco_file, image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up optimizer
    optimizer = torch.optim.Adam(predictor.model.parameters(), lr=lr)

    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor.model.to(device)

    for epoch in range(epochs):
        predictor.model.train()
        total_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # Ensure masks have the correct shape (add channel dimension if needed)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            # Wrap images into the expected format for SAM
            batched_input = [
                {
                    "image": img,  # Image tensor
                    "original_size": (img.shape[1], img.shape[2])  # Height, Width
                }
                for img in images
            ]

            # Forward pass with multimask_output argument
            outputs = predictor.model(batched_input, multimask_output=False)  # SAM output

            # Extract predictions from outputs and reshape
            predictions = torch.stack([output["masks"].squeeze(1) for output in outputs], dim=0)

            # Ensure predictions match the shape of masks
            if predictions.ndim == 3:
                predictions = predictions.unsqueeze(1)

            # Print shapes for debugging
            print(f"Predictions shape: {predictions.shape}")
            print(f"Masks shape: {masks.shape}")

            # Ensure shapes match exactly
            min_height = min(predictions.shape[2], masks.shape[2])
            min_width = min(predictions.shape[3], masks.shape[3])
            
            predictions = predictions[:, :, :min_height, :min_width]
            masks = masks[:, :, :min_height, :min_width]

            # Ensure masks are float and in the range [0, 1]
            masks = masks.float()
            masks = torch.clamp(masks, min=0, max=1)

            # Compute loss
            loss = criterion(predictions, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # Save the trained SAM model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(sam.state_dict(), os.path.join(output_dir, "sam_lora.pth"))
    print(f"LoRA-SAM model saved to {output_dir}")


if __name__ == "__main__":
    # Paths to the dataset and model checkpoint
    IMAGE_DIR = "dataset/small_datasets/Chafing/Chafing_High"  # Adjust to your image directory
    MODEL_CHECKPOINT = "sam_vit_b_01ec64.pth"  # Replace with the path to SAM checkpoint

    # Train the model
    train_lora_sam(COCO_DATASET_PATH, IMAGE_DIR, OUTPUT_DIR, MODEL_CHECKPOINT)