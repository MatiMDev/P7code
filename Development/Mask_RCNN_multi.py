import os
import cv2
import logging
from datetime import datetime
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Verify PyTorch CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Ensure your PyTorch installation supports CUDA.")
        raise RuntimeError("CUDA is required but not available.")

    logger.info(f"Using CUDA version: {torch.version.cuda}")
    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")

    # List of dataset folders
    DATASET_FOLDERS = ["CutStrands.v1i.coco-segmentation", "Plucking.v3i.coco-segmentation"]
    ANNOTATIONS_FILE_NAME = "_annotations.coco.json"

    # Configuration
    ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
    CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
    MAX_ITER = 2000
    EVAL_PERIOD = 200
    BASE_LR = 0.001
    NUM_CLASSES = 3  # Update if the number of classes differs
    OUTPUT_BASE_DIR = "output"
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Train on each dataset in the list
    for dataset_folder in DATASET_FOLDERS:
        dataset_name = os.path.basename(dataset_folder)
        logger.info(f"Processing dataset: {dataset_name}")

        # Dataset paths
        train_dir = os.path.join(dataset_folder, "train")
        valid_dir = os.path.join(dataset_folder, "valid")
        test_dir = os.path.join(dataset_folder, "test")

        # Register datasets
        train_name = f"{dataset_name}_train"
        valid_name = f"{dataset_name}_valid"
        test_name = f"{dataset_name}_test"

        register_coco_instances(train_name, {}, os.path.join(train_dir, ANNOTATIONS_FILE_NAME), train_dir)
        register_coco_instances(valid_name, {}, os.path.join(valid_dir, ANNOTATIONS_FILE_NAME), valid_dir)
        register_coco_instances(test_name, {}, os.path.join(test_dir, ANNOTATIONS_FILE_NAME), test_dir)

        # Configure the model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_PATH))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE_PATH)
        cfg.DATASETS.TRAIN = (train_name,)
        cfg.DATASETS.TEST = (valid_name,)
        cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
        cfg.DATALOADER.NUM_WORKERS = 0  # Prevent multiprocessing issues
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = BASE_LR
        cfg.SOLVER.MAX_ITER = MAX_ITER
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
        cfg.MODEL.DEVICE = "cuda"
        cfg.OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, dataset_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # Training
        logger.info(f"Starting training for dataset: {dataset_name}")
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        logger.info(f"Training completed for dataset: {dataset_name}")

        # Evaluation
        logger.info(f"Starting evaluation for dataset: {dataset_name}")
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        predictor = DefaultPredictor(cfg)

        metadata = MetadataCatalog.get(valid_name)
        dataset_valid = DatasetCatalog.get(valid_name)

        for d in dataset_valid:
            img = cv2.imread(d["file_name"])
            outputs = predictor(img)

            visualizer = Visualizer(
                img[:, :, ::-1],
                metadata=metadata,
                scale=0.8,
                instance_mode=ColorMode.IMAGE_BW
            )
            out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
            output_image_path = os.path.join(cfg.OUTPUT_DIR, os.path.basename(d["file_name"]))
            cv2.imwrite(output_image_path, out.get_image()[:, :, ::-1])
            logger.info(f"Processed and saved: {output_image_path}")

        logger.info(f"Evaluation completed for dataset: {dataset_name}")
