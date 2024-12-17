import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.processor import Samprocessor
from src.lora import LoRA_sam
from src.segment_anything import build_sam_vit_b
from src.utils import stacking_batch
from src.dataloader import COCOToDataset  # Ensure this is defined in your `src.dataloader` module
import yaml
import monai
import os

# Custom collate function to ensure batch is a list of dictionaries
def custom_collate_fn(batch):
    return list(batch)

# Load the config file
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("./lora_weights", exist_ok=True)  # Ensure output directory exists

# Initialize LoRA-SAM
print("Initializing LoRA-SAM...")
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])
processor = Samprocessor(sam_lora.sam)

# Dataset Preparation
print("Loading datasets...")
train_dataset = COCOToDataset(
    coco_json="./Plucking.v2i.coco-segmentation/train/_annotations.coco.json",
    image_root="./Plucking.v2i.coco-segmentation/train",
    processor=processor
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config_file["TRAIN"]["BATCH_SIZE"],
    shuffle=True,
    collate_fn=custom_collate_fn
)

valid_dataset = COCOToDataset(
    coco_json="./Plucking.v2i.coco-segmentation/valid/_annotations.coco.json",
    image_root="./Plucking.v2i.coco-segmentation/valid",
    processor=processor
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=config_file["TRAIN"]["BATCH_SIZE"],
    shuffle=False,
    collate_fn=custom_collate_fn
)

# Training Setup
print("Setting up training...")
model = sam_lora.sam
model.train().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

# Training Loop
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]
best_val_loss = float("inf")  # For tracking the best validation loss
checkpoint_path = f"./lora_weights/lora_rank{config_file['SAM']['RANK']}.safetensors"

print("Starting training...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = []
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        outputs = model(batched_input=batch, multimask_output=False)
        stk_gt, stk_out = stacking_batch(batch, outputs)
        stk_out = stk_out.squeeze(1)
        stk_gt = stk_gt.unsqueeze(1)

        loss = seg_loss(stk_out, stk_gt.float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    # Validation phase
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            outputs = model(batched_input=batch, multimask_output=False)
            stk_gt, stk_out = stacking_batch(batch, outputs)
            stk_out = stk_out.squeeze(1)
            stk_gt = stk_gt.unsqueeze(1)

            loss = seg_loss(stk_out, stk_gt.float().to(device))
            val_loss.append(loss.item())

    # Log the losses
    avg_train_loss = sum(train_loss) / len(train_loss)
    avg_val_loss = sum(val_loss) / len(val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Save the model if validation improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        sam_lora.save_lora_parameters(checkpoint_path)
        print(f"Saved best model with validation loss {best_val_loss:.4f} to {checkpoint_path}")

# Final Save
print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
print(f"Final model saved to {checkpoint_path}")
