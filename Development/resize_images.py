import cv2
import os
import json

def resize_images_and_annotations(image_input_dir, annotation_file, image_output_dir, annotation_output_file, original_width, original_height, target_size=(640, 640)):
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)

    scale_x = target_size[0] / original_width
    scale_y = target_size[1] / original_height

    # Load the annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    image_files = os.listdir(image_input_dir)
    total_files = len(image_files) - 1
    processed_files = 0

    for img_name in image_files:
        img_path = os.path.join(image_input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize the image
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(image_output_dir, img_name), resized_img)

        # Update the annotations
        for key, value in annotations.items():
            if value['filename'] == img_name:  # Match by filename
                for region in value['regions']:
                    if 'shape_attributes' in region:
                        region['shape_attributes']['all_points_x'] = [
                            int(x * scale_x) for x in region['shape_attributes']['all_points_x']
                        ]
                        region['shape_attributes']['all_points_y'] = [
                            int(y * scale_y) for y in region['shape_attributes']['all_points_y']
                        ]
                break

        # Update and display progress
        processed_files += 1
        progress = (processed_files / total_files) * 100
        print(f"Progress: {processed_files}/{total_files} images processed ({progress:.2f}%)", end='\r')

    print("\nProcessing complete.")

    # Save the updated annotations
    with open(annotation_output_file, 'w') as f:
        json.dump(annotations, f, indent=4)

# Define input and output paths
image_input_directory = "dataset_full/Chafing/Chafing/Chafing_High"
annotation_input_file = "dataset_full/Chafing/Chafing/Chafing_High/chafing_high_json.json"
image_output_directory = "dataset_full_scaled/Chafing/Chafing/Chafing_High"
annotation_output_file = "dataset_full_scaled/Chafing/Chafing/Chafing_High/annotations.json"

# Original dimensions of the images
original_width = 2000
original_height = 1080

# Target size for resizing
target_size = (640, 640)

resize_images_and_annotations(
    image_input_dir=image_input_directory,
    annotation_file=annotation_input_file,
    image_output_dir=image_output_directory,
    annotation_output_file=annotation_output_file,
    original_width=original_width,
    original_height=original_height,
    target_size=target_size
)
