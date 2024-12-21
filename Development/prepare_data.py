import json
import os

from pathlib import Path

def convert_to_yolo_style(input_file, output_folder, image_width=2000, image_height=1080):
    """
    Converts custom label format to YOLO-style annotations.

    Args:
        input_file (str): Path to the input label file.
        output_folder (str): Path to save YOLO-style annotations.
        image_width (int): Width of the images.
        image_height (int): Height of the images.
    """
    with open(input_file, 'r') as file:
        data = json.load(file)

    yolo_annotations = {}

    for key, value in data.items():
        filename = value['filename']
        regions = value.get('regions', [])
        yolo_data = []

        if not regions:
            print(f"Warning: No regions found for {filename}. Skipping.")
            continue

        for region_index, region in enumerate(regions):
            shape = region.get('shape_attributes', {})
            attributes = region.get('region_attributes', {})

            # Ensure the shape is a polygon and points are valid
            if shape.get('name') != 'polygon':
                print(f"Skipping non-polygon region in {filename}, region {region_index}.")
                continue
            if not shape.get('all_points_x') or not shape.get('all_points_y'):
                print(f"Missing points in {filename}, region {region_index}.")
                continue

            # Extract points
            x_points = shape['all_points_x']
            y_points = shape['all_points_y']

            # Calculate bounding box
            x_min, x_max = min(x_points), max(x_points)
            y_min, y_max = min(y_points), max(y_points)

            # Convert to YOLO format
            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            # Dynamically assign class ID (adjust if needed)
            class_id = attributes.get('class', 0)

            # Add annotation
            yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Add to annotations if regions exist
        if yolo_data:
            yolo_annotations[filename] = "\n".join(yolo_data)
        else:
            print(f"Warning: No valid regions for {filename}. Skipping.")

    # Save YOLO annotations
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + '_yolo.json')
    with open(output_file, 'w') as file:
        json.dump(yolo_annotations, file, indent=4)
    print(f"YOLO-style annotations saved to {output_file}")




def convert_to_coco_style(input_file, output_folder, image_width=2000, image_height=1080):
    """
    Converts a custom label format into COCO-style annotations.

    Args:
        input_file (str): Path to the input label file.
        output_folder (str): Path to save the COCO-style JSON file.
        image_width (int): Width of the images.
        image_height (int): Height of the images.
    """
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "object"}]  # Adjust categories as needed
    }
    annotation_id = 1  # Unique ID for each annotation
    image_id = 1       # Unique ID for each image

    for key, value in data.items():
        filename = value['filename']
        regions = value.get('regions', [])
        
        # Add image metadata
        coco_annotations["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": image_width,
            "height": image_height
        })

        for region in regions:
            shape = region['shape_attributes']
            attributes = region['region_attributes']
            if shape['name'] != 'polygon':
                continue  # Skip non-polygon shapes
            
            # Get all points for the polygon
            x_points = shape['all_points_x']
            y_points = shape['all_points_y']
            segmentation = [coord for pair in zip(x_points, y_points) for coord in pair]

            # Calculate bounding box
            x_min, x_max = min(x_points), max(x_points)
            y_min, y_max = min(y_points), max(y_points)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            # Add annotation
            coco_annotations["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 0,  # Adjust category ID as needed
                "segmentation": [segmentation],
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    # Save the COCO-style annotations to a new file in the output folder
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + '_coco.json')
    with open(output_file, 'w') as file:
        json.dump(coco_annotations, file, indent=4)
    print(f"COCO-style annotations saved to {output_file}")

def main(input_file, output_folder):
    """
    Main function to convert data into both YOLO and COCO formats.

    Args:
        input_file (str): Path to the input label file.
        output_folder (str): Path to the output folder for the converted files.
    """
    # Convert to YOLO format
    convert_to_yolo_style(input_file, output_folder)
    # Convert to COCO format
    convert_to_coco_style(input_file, output_folder)


# Example usage
if __name__ == "__main__":
    input_file_path = 'dataset\small_datasets\Chafing\Chafing_High\smaller_dataset_labels.json'  # Replace with the actual path
    output_folder_path = 'dataset\small_datasets\Chafing\Chafing_High'        # Replace with the desired output folder
    main(input_file_path, output_folder_path)