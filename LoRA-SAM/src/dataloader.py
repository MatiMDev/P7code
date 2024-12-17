import torch
import glob
import os 
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import src.utils as utils
import yaml

from pycocotools.coco import COCO
import torch
import numpy as np
from PIL import Image
from src.utils import get_bounding_box

class COCOToDataset(torch.utils.data.Dataset):
    def __init__(self, coco_json, image_root, processor):
        self.coco = COCO(coco_json)
        self.image_root = image_root
        self.processor = processor
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = f"{self.image_root}/{img_info['file_name']}"
        image = Image.open(img_path).convert("RGB")

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Create mask and bounding box
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))
        bbox = get_bounding_box(mask)

        # Process the image and bounding box
        original_size = (img_info["height"], img_info["width"])
        inputs = self.processor(image, original_size, bbox)
        inputs["ground_truth_mask"] = torch.from_numpy(mask)

        return inputs





class DatasetSegmentation(Dataset):
    """
    Dataset to process the images and masks

    Arguments:
        folder_path (str): The path of the folder containing the images
        processor (obj): Samprocessor class that helps pre processing the image, and prompt 
    
    Return:
        (dict): Dictionnary with 4 keys (image, original_size, boxes, ground_truth_mask)
            image: image pre processed to 1024x1024 size
            original_size: Original size of the image before pre processing
            boxes: bouding box after adapting the coordinates of the pre processed image
            ground_truth_mask: Ground truth mask
    """

    def __init__(self, config_file: dict, processor: Samprocessor, mode: str):
        super().__init__()
        if mode == "train":
            self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TRAIN_PATH"],'images','*.jpg'))
            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(config_file["DATASET"]["TRAIN_PATH"],'masks', os.path.basename(img_path)[:-4] + ".jpg")) 

        else:
            self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TEST_PATH"],'images','*.jpg'))
            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(config_file["DATASET"]["TEST_PATH"],'masks', os.path.basename(img_path)[:-4] + ".jpg"))


        self.processor = processor

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index: int) -> list:
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            # get image and mask in PIL format
            image =  Image.open(img_path)
            mask = Image.open(mask_path)
            mask = mask.convert('1')
            ground_truth_mask =  np.array(mask)
            original_size = tuple(image.size)[::-1]
    
            # get bounding box prompt
            box = utils.get_bounding_box(ground_truth_mask)
            inputs = self.processor(image, original_size, box)
            inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)

            return inputs
    
def collate_fn(batch):
    # Ensure the batch is returned as a list of dictionaries
    return list(batch)
