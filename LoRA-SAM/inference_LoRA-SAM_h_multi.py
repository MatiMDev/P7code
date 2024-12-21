import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from src.processor import Samprocessor
from src.lora import LoRA_sam
from src.segment_anything import build_sam_vit_h  # Changed from build_sam_vit_b
from src.dataloader import COCOToDataset
from src.utils import stacking_batch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
import os
import logging
import csv
import numpy as np
import re

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
with open("./config_h.yaml", "r") as ymlfile:
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
                stk_out = stk_out.squeeze(1)
                stk_gt = stk_gt.unsqueeze(1).float()

                # Calculate loss
                loss = seg_loss(stk_out, stk_gt.to(device))
                total_loss.append(loss.item())

                # Generate binary prediction mask
                pred_mask = (stk_out.sigmoid() > 0.5).float()

                # Calculate metrics
                iou = compute_iou(pred_mask, stk_gt)
                precision, recall, f1 = compute_precision_recall_f1(pred_mask, stk_gt)

                iou_list.append(iou)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

            except Exception as e:
                logger.error(f"Error during evaluation: {str(e)}")
                continue

    metrics = {
        "Mean Dice Loss": mean(total_loss) if total_loss else 0,
        "Mean IoU": mean(iou_list) if iou_list else 0,
        "Mean Precision": mean(precision_list) if precision_list else 0,
        "Mean Recall": mean(recall_list) if recall_list else 0,
        "Mean F1-Score": mean(f1_list) if f1_list else 0
    }

    logger.info(f"Detailed metrics: {metrics}")
    return metrics

# Custom collate function
def custom_collate_fn(batch):
    return list(batch)

def get_available_ranks(weights_dir="./lora_weights_h"):
    """Find all available LoRA weights and extract their ranks."""
    if not os.path.exists(weights_dir):
        return []
    
    ranks = []
    for file in os.listdir(weights_dir):
        if file.endswith('.safetensors'):
            match = re.search(r'rank(\d+)', file)
            if match:
                ranks.append(int(match.group(1)))
    return sorted(ranks)

def main():
    # Load the baseline SAM model (now using ViT-H)
    logger.info("Loading baseline SAM-H model...")
    sam = build_sam_vit_h(checkpoint=config_file["SAM"]["CHECKPOINT"])
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
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # Evaluate Baseline SAM
    print("Evaluating baseline SAM-H...")
    baseline_metrics = evaluate_model(sam, test_dataloader)
    print(f"Baseline Metrics: {baseline_metrics}")

    # Get all available ranks
    available_ranks = get_available_ranks()
    if not available_ranks:
        print("No LoRA weights found in ./lora_weights_h directory")
        return

    print(f"Found weights for ranks: {available_ranks}")
    
    # Store all results
    all_metrics = {"Baseline": baseline_metrics}

    # Evaluate each available rank
    for rank in available_ranks:
        print(f"\nEvaluating LoRA-SAM-H with Rank {rank}...")
        try:
            sam_lora = LoRA_sam(build_sam_vit_h(checkpoint=config_file["SAM"]["CHECKPOINT"]), rank)
            sam_lora.load_lora_parameters(f"./lora_weights_h/lora_rank{rank}.safetensors")
            rank_metrics = evaluate_model(sam_lora.sam, test_dataloader)
            all_metrics[f"Rank {rank}"] = rank_metrics
            print(f"Rank {rank} Metrics: {rank_metrics}")
        except Exception as e:
            print(f"Error evaluating rank {rank}: {str(e)}")
            logger.error(f"Error evaluating rank {rank}: {str(e)}")
            continue

    # Save all metrics to CSV
    os.makedirs("./output_data", exist_ok=True)
    csv_file = "./output_data/evaluation_results_sam_h_all_ranks.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dice Loss", "IoU", "Precision", "Recall", "F1-Score"])
        for model_name, metrics in all_metrics.items():
            writer.writerow([model_name, *metrics.values()])
    print(f"Metrics saved to {csv_file}")

    # Create comparison plots
    plt.figure(figsize=(15, 5))
    metrics_to_plot = ["Mean Dice Loss", "Mean IoU", "Mean F1-Score"]
    
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(1, 3, i)
        models = list(all_metrics.keys())
        values = [metrics[metric] for metrics in all_metrics.values()]
        
        plt.bar(range(len(models)), values)
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.title(metric)
        if metric == "Mean Dice Loss":
            plt.ylabel("Loss")
        else:
            plt.ylabel("Score")
        
        # Add value labels on top of each bar
        for j, v in enumerate(values):
            plt.text(j, v, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("./plots/comparison_plot_sam_h_all_ranks.jpg", bbox_inches='tight', dpi=300)
    print("Plot saved to ./plots/comparison_plot_sam_h_all_ranks.jpg")

    # Print summary of improvements
    baseline_iou = baseline_metrics["Mean IoU"]
    best_rank = max(
        [(rank, metrics["Mean IoU"]) for rank, metrics in all_metrics.items() if rank != "Baseline"],
        key=lambda x: x[1]
    )
    print("\nSummary:")
    print(f"Baseline IoU: {baseline_iou:.4f}")
    print(f"Best performance: {best_rank[0]} (IoU: {best_rank[1]:.4f})")
    print(f"Improvement: {((best_rank[1] - baseline_iou) / baseline_iou * 100):.2f}%")

if __name__ == "__main__":
    main()