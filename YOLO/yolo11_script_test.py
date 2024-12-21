import os
import cv2
from pathlib import Path
try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    raise ImportError("The 'ultralytics' library is required but not installed. Please install it using 'pip install ultralytics' and try again.")

# Initialize the YOLO model
model_path = os.path.join('runs', 'segment', 'train', 'weights', 'best.pt')
model = YOLO(model_path)

# Define the folder containing images
image_folder = "dataset_yolo/valid/images/"

# Get a list of all image files in the folder
image_files = list(Path(image_folder).glob("*.jpg"))

if not image_files:
    raise FileNotFoundError(f"No .jpg files found in folder: {image_folder}")

# Process each image in the folder
for image_file in image_files:
    print(f"Processing image: {image_file}")
    results = model.predict(
        source=str(image_file),
        imgsz=640,
        conf=0.5,
        save=False
    )

    for result in results:
        # Plot the results as an overlay on the image
        overlay = result.plot()
        cv2.imshow('Result', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print("Processing complete.")