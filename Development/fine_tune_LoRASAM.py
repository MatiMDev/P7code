import os
import cv2
import logging
from datetime import datetime
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from segment_anything import sam_model_registry, SamPredictor
from Sam_LoRA.sam_lora import LoRA_Sam

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verify PyTorch CUDA availability
if not torch.cuda.is_available():
    logger.error("CUDA is not available. Ensure your PyTorch installation supports CUDA.")
    raise RuntimeError("CUDA is required but not available.")

logger.info(f"Using CUDA version: {torch.version.cuda}")
logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")

# Dataset configuration
DATASET_PATH = "Plucking.v2i.coco-segmentation"
ANNOTATIONS_FILE_NAME = "_annotations.coco.json"
TRAIN_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATASET_PATH, "train")
VALID_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATASET_PATH, "valid")
TEST_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATASET_PATH, "test")

# Load COCO-style annotations
def load_coco_annotations(annotation_file):
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    with open(annotation_file, 'r') as f:
        return json.load(f)

logger.info("Loading annotations for each dataset split...")
train_annotations = load_coco_annotations(os.path.join(TRAIN_DATA_SET_IMAGES_DIR_PATH, ANNOTATIONS_FILE_NAME))
valid_annotations = load_coco_annotations(os.path.join(VALID_DATA_SET_IMAGES_DIR_PATH, ANNOTATIONS_FILE_NAME))
test_annotations = load_coco_annotations(os.path.join(TEST_DATA_SET_IMAGES_DIR_PATH, ANNOTATIONS_FILE_NAME))
logger.info("Annotations loaded successfully for train, valid, and test datasets.")

# Custom dataset for LoRA-SAM
class COCODataset(Dataset):
    def __init__(self, images_dir, annotations):
        self.images_dir = images_dir
        self.annotations = annotations["images"]
        self.masks = annotations["annotations"]
        self.image_id_to_masks = {ann["image_id"]: ann for ann in self.masks}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_info = self.annotations[idx]
        image_path = os.path.join(self.images_dir, image_info["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_id = image_info["id"]

        # Generate mask
        mask_info = self.image_id_to_masks.get(image_id, {})
        mask = torch.zeros((image.shape[0], image.shape[1]), dtype=torch.float32)
        if mask_info:
            for seg in mask_info.get("segmentation", []):
                poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask.numpy(), [poly], 1)

        return torch.tensor(image).permute(2, 0, 1) / 255.0, mask.unsqueeze(0)

# Load SAM and LoRA-SAM
logger.info("Loading SAM and wrapping with LoRA...")
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
lora_sam = LoRA_Sam(sam, r=8)
predictor = SamPredictor(lora_sam.sam)
predictor.model = predictor.model.to("cuda")  # Move SAM model to GPU
logger.info("LoRA-SAM loaded successfully.")

def fine_tune_model(model, train_loader, valid_loader, num_epochs=10, lr=1e-4):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to("cuda").float()  # Ensure images are float
            masks = masks.to("cuda").squeeze(1).float()  # Ensure masks are float

            # Prepare input for SAM
            batched_input = [{"image": image, "original_size": (image.shape[1], image.shape[2])} for image in images]

            optimizer.zero_grad()
            outputs = model(batched_input, multimask_output=False)  # Proper SAM input format

            # Collect masks from outputs while preserving the graph
            predicted_masks = torch.stack([out["masks"][0] for out in outputs], dim=0)

            # Ensure outputs have the same shape as masks
            predicted_masks = predicted_masks.float()  # Ensure predicted masks are float
            loss = criterion(predicted_masks, masks)  # Compute the loss
            loss.backward()  # Backpropagate
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.to("cuda").float()  # Ensure images are float
                masks = masks.to("cuda").squeeze(1).float()  # Ensure masks are float

                # Prepare input for SAM
                batched_input = [{"image": image, "original_size": (image.shape[1], image.shape[2])} for image in images]
                outputs = model(batched_input, multimask_output=False)

                # Collect masks from outputs
                predicted_masks = torch.stack([out["masks"][0] for out in outputs], dim=0)

                # Ensure outputs have the same shape as masks
                predicted_masks = predicted_masks.float()  # Ensure predicted masks are float
                loss = criterion(predicted_masks, masks)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Valid Loss: {valid_loss:.4f}")

# Data loaders
train_dataset = COCODataset(TRAIN_DATA_SET_IMAGES_DIR_PATH, train_annotations)
valid_dataset = COCODataset(VALID_DATA_SET_IMAGES_DIR_PATH, valid_annotations)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

logger.info("Starting fine-tuning...")
fine_tune_model(lora_sam.sam, train_loader, valid_loader, num_epochs=10, lr=1e-4)
logger.info("Fine-tuning completed.")

# Final evaluation
FINAL_OUTPUT_DIR = os.path.join(DATASET_PATH, "lora_sam_outputs", datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

def evaluate_and_visualize(images_dir, annotations, predictor, output_dir):
    for image_info in annotations["images"]:
        image_path = os.path.join(images_dir, image_info["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform prediction
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            multimask_output=True
        )

        # Save final visualization
        overlay = image.copy()
        for mask in masks:
            overlay[mask] = [0, 255, 0]  # Green mask overlay
        combined = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
        output_path = os.path.join(output_dir, f"{os.path.basename(image_path)}")
        cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved final visualization: {output_path}")

logger.info("Starting evaluation on test dataset...")
evaluate_and_visualize(TEST_DATA_SET_IMAGES_DIR_PATH, test_annotations, predictor, FINAL_OUTPUT_DIR)
logger.info("Evaluation and visualization completed.")
