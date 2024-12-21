import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from src.processor import Samprocessor
from src.lora import LoRA_sam
from src.segment_anything import build_sam_vit_b
from src.dataloader import COCOToDataset
from src.utils import stacking_batch  # Make sure to import stacking_batch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
import os
import logging
import csv
import numpy as np

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "evaluation_log.txt"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Loss function
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

# Load configuration
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Function to compute IoU
def compute_iou(pred_mask, gt_mask):
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    intersection = (pred_mask & gt_mask).float().sum()
    union = (pred_mask | gt_mask).float().sum()
    return (intersection / union).item() if union > 0 else 0.0

# Function to compute Precision, Recall, and F1-score
def compute_precision_recall_f1(pred_mask, gt_mask):
    pred_flat = pred_mask.bool().cpu().numpy().flatten()
    gt_flat = gt_mask.bool().cpu().numpy().flatten()
    
    precision = precision_score(gt_flat, pred_flat, average="binary", zero_division=0)
    recall = recall_score(gt_flat, pred_flat, average="binary", zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, average="binary", zero_division=0)
    return precision, recall, f1

# Function to evaluate a model
def evaluate_model(model, dataloader):
    model.eval().to(device)
    total_loss = []
    iou_list, precision_list, recall_list, f1_list = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                # Move input data to device
                for item in batch:
                    for k, v in item.items():
                        if isinstance(v, torch.Tensor):
                            item[k] = v.to(device)

                # Get model predictions
                outputs = model(batched_input=batch, multimask_output=False)
                
                # Process ground truth and predictions
                stk_gt, stk_out = stacking_batch(batch, outputs)
                stk_out = stk_out.squeeze(1)  # Remove unnecessary dimensions
                stk_gt = stk_gt.unsqueeze(1).float()  # Add channel dimension and convert to float

                # Calculate loss
                loss = seg_loss(stk_out, stk_gt.to(device))
                total_loss.append(loss.item())

                # Generate binary prediction mask
                pred_mask = (stk_out.sigmoid() > 0.5).float()  # Adjusted threshold to 0.5

                # Calculate metrics
                iou = compute_iou(pred_mask, stk_gt)
                precision, recall, f1 = compute_precision_recall_f1(pred_mask, stk_gt)

                iou_list.append(iou)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

                # Log some information for debugging
                if len(total_loss) == 1:  # Only for first batch
                    logger.info(f"First batch metrics - Loss: {loss.item():.4f}, IoU: {iou:.4f}")
                    logger.info(f"Prediction range: {pred_mask.min():.4f} to {pred_mask.max():.4f}")
                    logger.info(f"Ground truth range: {stk_gt.min():.4f} to {stk_gt.max():.4f}")

            except Exception as e:
                logger.error(f"Error during evaluation: {str(e)}")
                continue

    # Calculate mean metrics
    metrics = {
        "Mean Dice Loss": mean(total_loss) if total_loss else 0,
        "Mean IoU": mean(iou_list) if iou_list else 0,
        "Mean Precision": mean(precision_list) if precision_list else 0,
        "Mean Recall": mean(recall_list) if recall_list else 0,
        "Mean F1-Score": mean(f1_list) if f1_list else 0
    }

    # Log detailed metrics
    logger.info(f"Detailed metrics: {metrics}")
    return metrics

# Custom collate function
def custom_collate_fn(batch):
    return list(batch)

def main():
    # Load the baseline SAM model
    logger.info("Loading baseline SAM model...")
    sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
    processor = Samprocessor(sam)

    # Load the test dataset
    logger.info("Loading test dataset...")
    test_dataset = COCOToDataset(
        coco_json="./Plucking.v2i.coco-segmentation/test/_annotations.coco.json",
        image_root="./Plucking.v2i.coco-segmentation/test",
        processor=processor
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,  # Use batch size 1 for evaluation
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # Evaluate Baseline SAM
    print("Evaluating baseline SAM...")
    baseline_metrics = evaluate_model(sam, test_dataloader)
    print(f"Baseline Metrics: {baseline_metrics}")

    # Only evaluate the rank specified in config
    rank = config_file["SAM"]["RANK"]
    print(f"Evaluating LoRA-SAM with Rank {rank}...")
    try:
        sam_lora = LoRA_sam(build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"]), rank)
        sam_lora.load_lora_parameters(f"./lora_weights/lora_rank{rank}.safetensors")
        lora_metrics = evaluate_model(sam_lora.sam, test_dataloader)
        print(f"Rank {rank} Metrics: {lora_metrics}")
        
        # Save metrics to CSV
        os.makedirs("./output_data", exist_ok=True)
        csv_file = "./output_data/evaluation_results.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Dice Loss", "IoU", "Precision", "Recall", "F1-Score"])
            writer.writerow(["Baseline", *baseline_metrics.values()])
            writer.writerow([f"LoRA Rank {rank}", *lora_metrics.values()])
        print(f"Metrics saved to {csv_file}")

        # Plot results
        metrics_names = ["Dice Loss", "IoU", "F1-Score"]
        baseline_values = [baseline_metrics[f"Mean {name}"] for name in metrics_names]
        lora_values = [lora_metrics[f"Mean {name}"] for name in metrics_names]

        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics_names))
        width = 0.35

        plt.bar(x - width/2, baseline_values, width, label='Baseline SAM')
        plt.bar(x + width/2, lora_values, width, label=f'LoRA-SAM (Rank {rank})')

        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Comparison of Baseline SAM vs LoRA-SAM')
        plt.xticks(x, metrics_names)
        plt.legend()

        os.makedirs("./plots", exist_ok=True)
        plt.savefig("./plots/comparison_plot.jpg")
        print("Plot saved to ./plots/comparison_plot.jpg")

    except FileNotFoundError:
        print(f"Rank {rank}: LoRA weight file not found.")
        logger.error(f"LoRA weight file not found for rank {rank}")

if __name__ == "__main__":
    main()