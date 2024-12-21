import os
import zipfile
import random
import cv2
import numpy as np
from pathlib import Path
try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    raise ImportError("The 'ultralytics' library is required but not installed. Please install it using 'pip install ultralytics' and try again.")
import matplotlib.pyplot as plt

# Unzip the dataset
zip_file_name = 'datasets/yolo_data.zip'
extract_folder = 'datasets/dataset_yolo'
os.makedirs(extract_folder, exist_ok=True)
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)
print(f'Extracted files to: {extract_folder}')

# Paths to data
image_folder = os.path.join(extract_folder, 'train/images')
label_folder = os.path.join(extract_folder, 'train/labels')
validation_folder = os.path.join(extract_folder, 'valid/images')
data_yaml_path = os.path.join(extract_folder, 'data.yaml')

if not os.path.exists(validation_folder):
    os.makedirs(validation_folder)
    print(f"Validation folder '{validation_folder}' was missing and has been created.")

# Ensure validation folder is not empty
validation_images = os.listdir(validation_folder)
if not validation_images:
    # Copy some images from train/images to valid/images as placeholders
    train_images = os.listdir(image_folder)
    num_to_copy = min(5, len(train_images))
    for i in range(num_to_copy):
        src = os.path.join(image_folder, train_images[i])
        dst = os.path.join(validation_folder, train_images[i])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
    print(f"Populated validation folder with {num_to_copy} images from training data.")

# Check and create data.yaml if missing
if not os.path.exists(data_yaml_path):
    with open(data_yaml_path, 'w') as f:
        f.write(
            f"""train: {os.path.join(extract_folder, 'train/images')}
valid: {os.path.join(extract_folder, 'valid/images')}

nc: 1
names: ['class0']"""
        )
    print(f"Generated missing 'data.yaml' at {data_yaml_path}.")

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

image_files = os.listdir(image_folder)
image_files = [f for f in image_files if f.endswith('.jpg')]
label_files = [f.replace('.jpg', '.txt') for f in image_files]
matching_files = [(img, lbl) for img, lbl in zip(image_files, label_files) if os.path.exists(os.path.join(label_folder, lbl))]
selected_pairs = random.sample(matching_files, min(6, len(matching_files)))

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

model = YOLO('yolo11n-seg.pt')
results = model.train(
    data=data_yaml_path,
    epochs=10,
    imgsz=640,
    batch=8,
)

model = YOLO(os.path.join('runs', 'segment', 'train', 'weights', 'best.pt'))
results = model.predict(
    source=os.path.join(image_folder, 'sample_image.jpg'),
    imgsz=640,
    conf=0.5,
    save=False
)

for result in results:
    overlay = result.plot()
    cv2.imshow('Result', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
