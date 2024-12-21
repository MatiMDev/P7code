import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", 'git+https://github.com/facebookresearch/detectron2.git'])

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

# Verify PyTorch CUDA availability
if not torch.cuda.is_available():
    logger.error("CUDA is not available. Ensure your PyTorch installation supports CUDA.")
    raise RuntimeError("CUDA is required but not available.")

logger.info(f"Using CUDA version: {torch.version.cuda}")
logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")

# Dataset configuration
DATA_SET_NAME = "datasets"
dataset_location = "datasets"
ANNOTATIONS_FILE_NAME = "_annotations.coco.json"
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"
TEST_DATA_SET_NAME = f"{DATA_SET_NAME}-test"
VALID_DATA_SET_NAME = f"{DATA_SET_NAME}-valid"

# Dataset paths
TRAIN_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset_location, "train")
TEST_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset_location, "test")
VALID_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset_location, "valid")

# Register datasets
logger.info("Registering datasets...")
register_coco_instances(
    name=TRAIN_DATA_SET_NAME,
    metadata={},
    json_file=os.path.join(TRAIN_DATA_SET_IMAGES_DIR_PATH, ANNOTATIONS_FILE_NAME),
    image_root=TRAIN_DATA_SET_IMAGES_DIR_PATH
)
register_coco_instances(
    name=TEST_DATA_SET_NAME,
    metadata={},
    json_file=os.path.join(TEST_DATA_SET_IMAGES_DIR_PATH, ANNOTATIONS_FILE_NAME),
    image_root=TEST_DATA_SET_IMAGES_DIR_PATH
)
register_coco_instances(
    name=VALID_DATA_SET_NAME,
    metadata={},
    json_file=os.path.join(VALID_DATA_SET_IMAGES_DIR_PATH, ANNOTATIONS_FILE_NAME),
    image_root=VALID_DATA_SET_IMAGES_DIR_PATH
)
logger.info("Datasets registered successfully.")

# Configuration
logger.info("Setting up model configuration...")
ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
MAX_ITER = 16000
EVAL_PERIOD = 200
BASE_LR = 0.001
NUM_CLASSES = 3  # Update if the number of classes differs
OUTPUT_DIR_PATH = os.path.join(
    DATA_SET_NAME,
    ARCHITECTURE,
    datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
)
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_PATH))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE_PATH)
cfg.DATASETS.TRAIN = (TRAIN_DATA_SET_NAME,)
cfg.DATASETS.TEST = (TEST_DATA_SET_NAME,)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
cfg.DATALOADER.NUM_WORKERS = 1  # Reduce to avoid worker crashes
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.INPUT.MASK_FORMAT = 'bitmask'
cfg.SOLVER.BASE_LR = BASE_LR
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
cfg.MODEL.DEVICE = "cuda"
cfg.OUTPUT_DIR = OUTPUT_DIR_PATH
logger.info("Configuration setup complete.")

if __name__ == "__main__":
    # Training
    logger.info("Starting training...")
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    logger.info("Training completed.")

    # Evaluation
    logger.info("Starting evaluation...")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get(VALID_DATA_SET_NAME)
    dataset_valid = DatasetCatalog.get(VALID_DATA_SET_NAME)

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
        output_image_path = os.path.join(OUTPUT_DIR_PATH, os.path.basename(d["file_name"]))
        cv2.imwrite(output_image_path, out.get_image()[:, :, ::-1])
        logger.info(f"Processed and saved: {output_image_path}")

    logger.info("Evaluation completed.")
