#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install ultralytics')


# #import library

# In[4]:


import os
import cv2
import random
import zipfile
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt


# In[5]:


zip_file_name = '/content/aa.v1i.yolov11.zip'
extract_folder = 'dataset_yolo'
os.makedirs(extract_folder, exist_ok=True)
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)
print(f'Extracted files to: {extract_folder}')


# # Paths to data

# In[6]:


image_folder = "/content/dataset_yolo/train/images/"
label_folder = "/content/dataset_yolo/train/labels/"


# 
# # Function to visualize an image with its segmentation annotation

# In[7]:


def visualize_image_with_annotation(image_path, label_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        cls, *points = data
        points = [float(x) for x in points]
        points = [(int(points[i] * w), int(points[i + 1] * h)) for i in range(0, len(points), 2)]

        points = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)

    return image


# # Get all image files

# In[8]:


image_files = os.listdir(image_folder)
image_files = [f for f in image_files if f.endswith('.jpg')]
label_files = [f.replace('.jpg', '.txt') for f in image_files]
matching_files = [(img, lbl) for img, lbl in zip(image_files, label_files) if os.path.exists(os.path.join(label_folder, lbl))]
selected_pairs = random.sample(matching_files, min(6, len(matching_files)))


# #Create a grid of images

# In[9]:


num_images = len(selected_pairs)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i, (image_name, label_name) in enumerate(selected_pairs):
    image_path = os.path.join(image_folder, image_name)
    label_path = os.path.join(label_folder, label_name)
    annotated_image = visualize_image_with_annotation(image_path, label_path)
    axs[i // 3, i % 3].imshow(annotated_image)
    axs[i // 3, i % 3].axis("off")

plt.tight_layout()
plt.show()


# #Train the model

# In[10]:


model = YOLO('yolo11n-seg.pt')
results = model.train(
    data="/content/dataset_yolo/data.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
)


# In[16]:


from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow

model = YOLO('/content/runs/segment/train/weights/best.pt')
results = model.predict(source='/content/dataset_yolo/valid/images/compress-slid_66_png.rf.e1b11954781163c0421bcb8318f2b31d.jpg', imgsz=640, conf=0.5, save=False)

for result in results:
    overlay = result.plot()
    cv2_imshow(overlay)


# %%
