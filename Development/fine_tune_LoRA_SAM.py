import torch
import torch.nn as nn
from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, images, masks, processor):
        """
        Custom dataset for segmentation tasks
        
        Args:
            images (list): List of image paths or image arrays
            masks (list): Corresponding segmentation masks
            processor (SamProcessor): SAM image processor
        """
        self.images = images
        self.masks = masks
        self.processor = processor
        
        # Optional data augmentation
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(width=1024, height=1024)
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = self.load_image(self.images[idx])
        mask = self.load_mask(self.masks[idx])
        
        # Apply augmentations
        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
        
        # Prepare inputs for SAM
        inputs = self.processor(
            image,
            input_boxes=[[0, 0, image.shape[1], image.shape[0]]],  # Full image box
            return_tensors="pt"
        )
        
        # Create ground truth segmentation
        ground_truth = torch.tensor(mask).long()
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'input_boxes': inputs['input_boxes'].squeeze(),
            'ground_truth': ground_truth
        }

def load_sam_model(pretrained_model_name="facebook/sam-vit-huge"):
    """
    Load pre-trained SAM model and prepare for LoRA fine-tuning
    
    Args:
        pretrained_model_name (str): Hugging Face model identifier
    
    Returns:
        model with LoRA adaptations
    """
    # Load pre-trained SAM model
    model = SamModel.from_pretrained(pretrained_model_name)
    processor = SamProcessor.from_pretrained(pretrained_model_name)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank of low-rank adaptation
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Modules to adapt
        lora_dropout=0.1,
        bias="none"
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    return model, processor

def train_sam_lora(model, train_dataloader, val_dataloader, epochs=5, learning_rate=1e-4):
    """
    Fine-tune SAM with LoRA
    
    Args:
        model: SAM model with LoRA
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Optimization learning rate
    """
    # Prepare training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for batch in train_dataloader:
            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            input_boxes = batch['input_boxes'].to(device)
            ground_truth = batch['ground_truth'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                pixel_values=pixel_values, 
                input_boxes=input_boxes
            )
            
            # Compute loss
            loss = criterion(outputs.logits, ground_truth)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                pixel_values = batch['pixel_values'].to(device)
                input_boxes = batch['input_boxes'].to(device)
                ground_truth = batch['ground_truth'].to(device)
                
                outputs = model(
                    pixel_values=pixel_values, 
                    input_boxes=input_boxes
                )
                
                val_loss = criterion(outputs.logits, ground_truth)
                total_val_loss += val_loss.item()
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {total_train_loss/len(train_dataloader):.4f}")
        print(f"Val Loss: {total_val_loss/len(val_dataloader):.4f}")
    
    return model

def main():
    # Example usage
    # Replace with your actual image and mask paths
    image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
    mask_paths = ['path/to/mask1.png', 'path/to/mask2.png']
    
    # Load model and processor
    model, processor = load_sam_model()
    
    # Create dataset
    dataset = SegmentationDataset(image_paths, mask_paths, processor)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4)
    
    # Fine-tune model
    fine_tuned_model = train_sam_lora(model, train_dataloader, val_dataloader)
    
    # Save fine-tuned model
    fine_tuned_model.save_pretrained("sam_lora_finetuned")

if __name__ == "__main__":
    main()