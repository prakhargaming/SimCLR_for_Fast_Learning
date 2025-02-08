import os
import cv2
import numpy as np
from pathlib import Path
import shutil

def create_directory_structure(base_path):
    """Create the directory structure for the extracted symbols."""
    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            path = os.path.join(base_path, split, subdir)
            os.makedirs(path, exist_ok=True)

def extract_bbox(image, x_center, y_center, width, height):
    """Extract a bounding box from an image using YOLO format coordinates."""
    img_height, img_width = image.shape[:2]
    
    # Convert YOLO coordinates to pixel coordinates
    x_center = float(x_center) * img_width
    y_center = float(y_center) * img_height
    width = float(width) * img_width
    height = float(height) * img_height
    
    # Calculate bbox coordinates
    x1 = int(x_center - width/2)
    y1 = int(y_center - height/2)
    x2 = int(x_center + width/2)
    y2 = int(y_center + height/2)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    
    # Extract the region
    return image[y1:y2, x1:x2]

def process_dataset(input_base_path, output_base_path):
    """Process the entire dataset and extract symbols."""
    # Create output directory structure
    create_directory_structure(output_base_path)
    
    # Process each split (train/valid/test)
    for split in ['train', 'valid', 'test']:
        print(f"Processing {split} split...")
        
        images_dir = os.path.join(input_base_path, split, 'images')
        labels_dir = os.path.join(input_base_path, split, 'labels')
        
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in image_files:
            # Load image
            img_path = os.path.join(images_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not read image: {img_path}")
                continue
                
            # Get corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            if not os.path.exists(label_path):
                print(f"No label file found for: {img_file}")
                continue
            
            # Read annotations
            with open(label_path, 'r') as f:
                annotations = f.readlines()
            
            # Process each symbol in the image
            for idx, ann in enumerate(annotations):
                parts = ann.strip().split()
                if len(parts) >= 5:  # Ensure we have class and bbox coordinates
                    class_id = parts[0]
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Extract the symbol
                    symbol = extract_bbox(image, x_center, y_center, width, height)
                    
                    if symbol.size == 0:
                        print(f"Warning: Empty symbol extracted from {img_file}")
                        continue
                    
                    # Save the extracted symbol
                    symbol_filename = f"{os.path.splitext(img_file)[0]}_symbol_{idx}.png"
                    symbol_path = os.path.join(output_base_path, split, 'images', symbol_filename)
                    cv2.imwrite(symbol_path, symbol)
                    
                    # Create corresponding label file with class information
                    label_output_path = os.path.join(output_base_path, split, 'labels', 
                                                   f"{os.path.splitext(symbol_filename)[0]}.txt")
                    with open(label_output_path, 'w') as f:
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")  # Center the symbol in new annotation

def main():
    # Define input and output paths
    input_dataset_path = "datasets"
    output_dataset_path = "datasets_extracted_symbols"
    
    # Process the dataset
    process_dataset(input_dataset_path, output_dataset_path)
    print("Symbol extraction complete!")

if __name__ == "__main__":
    main()