import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from torchvision import transforms
import numpy as np  # Import numpy for array conversion
from PIL import ImageDraw  # Add ImageDraw to the imports

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoraSAMDataset(Dataset):
    def __init__(self, images_dir, annotations_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        # Load COCO-style annotations
        with open(annotations_path, "r") as f:
            data = json.load(f)
        
        self.images = data['images']
        self.annotations = data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in data['categories']}

        # Map annotations to images
        self.image_to_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image info
        img_info = self.images[idx]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get annotations and generate segmentation mask
        annotations = self.image_to_annotations.get(img_info['id'], [])
        mask = self.create_segmentation_mask(img_info['height'], img_info['width'], annotations)

        return {
            "image": image,
            "mask": mask,
            "annotations": annotations,
            "image_id": img_info['id']
        }

    def create_segmentation_mask(self, height, width, annotations):
        """
        Creates a binary segmentation mask for all objects in an image.
        """
        mask = torch.zeros((height, width), dtype=torch.uint8)

        for ann in annotations:
            segmentation = ann.get('segmentation', [])
            for polygon in segmentation:
                # Create a blank grayscale mask
                poly_mask = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(poly_mask)  # Initialize ImageDraw
                draw.polygon(polygon, outline=1, fill=1)  # Draw the polygon
                
                # Convert the poly_mask (PIL.Image) to a NumPy array and then to a PyTorch tensor
                poly_mask_tensor = torch.from_numpy(np.array(poly_mask, dtype=np.uint8))
                mask |= poly_mask_tensor  # Combine the mask

        return mask



def load_lora_sam_data(dataset_dir, transform=None):
    """
    Automatically loads train, valid, and test datasets from the specified directory.
    Assumes COCO-style annotations and folder structure:
        dataset_dir/
            train/
                _annotations.coco.json
                <images>
            valid/
                _annotations.coco.json
                <images>
            test/
                _annotations.coco.json
                <images>
    """
    splits = ["train", "valid", "test"]
    datasets = {}
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        annotations_path = os.path.join(split_dir, "_annotations.coco.json")
        if not os.path.exists(annotations_path):
            logger.warning(f"Annotations file not found for {split}. Skipping...")
            continue
        datasets[split] = LoraSAMDataset(
            images_dir=split_dir,
            annotations_path=annotations_path,
            transform=transform
        )
        logger.info(f"{split.capitalize()} dataset loaded with {len(datasets[split])} samples.")

    return datasets

# Transform for images
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# Dataset location
dataset_location = "Plucking.v2i.coco-segmentation"  # Update with the dataset folder path

# Load datasets
datasets = load_lora_sam_data(dataset_location, transform)

# Create dataloaders
batch_size = 4
dataloaders = {
    split: DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=2)
    for split, dataset in datasets.items()
}

# Example usage
if __name__ == "__main__":
    for split, dataloader in dataloaders.items():
        logger.info(f"Iterating through {split} dataloader...")
        for batch in dataloader:
            images, masks = batch["image"], batch["mask"]
            logger.info(f"{split.capitalize()} - Batch of images shape: {images.shape}")
            logger.info(f"{split.capitalize()} - Batch of masks shape: {masks.shape}")
            break
