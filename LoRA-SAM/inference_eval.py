import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from src.processor import Samprocessor
from src.dataloader import DatasetSegmentation, collate_fn
from src.lora import LoRA_sam
from src.segment_anything import build_sam_vit_b
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Loss function
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

# Load configuration
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Ranks to compare
rank_list = [2, 4, 6, 8, 16, 32, 64, 128, 256, 512]

# Directory to save plots
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

# Baseline and rank losses
baseline_loss = []
rank_loss = []

def evaluate_model(model, dataloader):
    model.eval().to(device)
    total_loss = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                outputs = model(batched_input=batch, multimask_output=False)
                gt_mask_tensor = batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0)
                loss = seg_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))
                total_loss.append(loss.item())
            except Exception as e:
                print(f"Error during evaluation: {e}")
                print(f"Batch: {batch}")
    
    if not total_loss:
        print("No loss was computed. Ensure your dataset and model are set up correctly.")
        return 0

    return mean(total_loss)


# Load the baseline SAM model
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
processor = Samprocessor(sam)

# Load the test dataset
test_dataset = DatasetSegmentation(config_file, processor, mode="test")
test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

print("Evaluating baseline SAM...")
baseline_loss = evaluate_model(sam, test_dataloader)
print(f"Baseline Mean Dice Loss: {baseline_loss:.4f}")

# Evaluate LoRA-SAM models for different ranks
for rank in rank_list:
    print(f"Evaluating LoRA-SAM with Rank {rank}...")
    # Create LoRA-SAM model
    sam_lora = LoRA_sam(build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"]), rank)
    sam_lora.load_lora_parameters(f"./lora_weights/lora_rank{rank}.safetensors")
    # Evaluate and store loss
    rank_loss.append(evaluate_model(sam_lora.sam, test_dataloader))
    print(f"Rank {rank} Mean Dice Loss: {rank_loss[-1]:.4f}")

# Plot results
print("Plotting results...")
models_results = {"Baseline": baseline_loss, **{f"Rank {rank}": loss for rank, loss in zip(rank_list, rank_loss)}}
x = np.arange(len(models_results))
width = 0.6

fig, ax = plt.subplots(figsize=(12, 6))
rects = ax.bar(x, models_results.values(), width, color="skyblue")
ax.bar_label(rects, padding=3)

# Add labels and title
ax.set_ylabel("Dice Loss", fontsize=12)
ax.set_title("LoRA-SAM Rank Comparison on Test Set", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models_results.keys(), rotation=45, ha="right", fontsize=10)
ax.set_ylim(0, max(models_results.values()) + 0.05)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(output_dir, "rank_comparison.jpg")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")