"""
Script to convert CSV files back to images.
Reads training_set.csv and test_set.csv and reconstructs 64x64 RGB images from the pixel data.
"""

import numpy as np
import pandas as pd
from PIL import Image
import os


def csv_to_images(csv_path, output_dir):
    """
    Convert CSV file with flattened image data back to PNG images.
    
    Args:
        csv_path: Path to the CSV file
        output_dir: Directory where images will be saved
    """
    # Read CSV
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Get labels and features
    labels = df['y'].values
    features = df.drop('y', axis=1).values
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    num_images = len(features)
    print(f"Found {num_images} images")
    
    for idx in range(num_images):
        # Get flattened pixel data
        flattened_pixels = features[idx]
        label = labels[idx]
        
        # Reshape from 12288 to (64, 64, 3)
        # The data is organized row by row, with RGB values for each pixel
        image_array = flattened_pixels.reshape((64, 64, 3))
        
        # Scale back to 0-255 range (assuming values are in 0-1 range)
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
        
        # Create image
        image = Image.fromarray(image_array, 'RGB')
        
        # Save image with label prefix
        label_name = "cat" if label == 1 else "non_cat"
        filename = f"{label_name}_{idx:04d}.png"
        output_path = os.path.join(output_dir, filename)
        image.save(output_path)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{num_images} images")
    
    print(f"Done! Images saved to {output_dir}")


def main():
    """Main function to process both training and test sets."""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process training set
    print("=" * 60)
    print("Processing Training Set")
    print("=" * 60)
    training_csv = os.path.join(script_dir, "training_set.csv")
    training_output = os.path.join(script_dir, "training_images")
    csv_to_images(training_csv, training_output)
    
    print("\n" + "=" * 60)
    print("Processing Test Set")
    print("=" * 60)
    # Process test set
    test_csv = os.path.join(script_dir, "test_set.csv")
    test_output = os.path.join(script_dir, "test_images")
    csv_to_images(test_csv, test_output)
    
    print("\n" + "=" * 60)
    print("All Done!")
    print("=" * 60)
    print(f"\nImages have been saved to:")
    print(f"  - Training: {training_output}")
    print(f"  - Test: {test_output}")


if __name__ == "__main__":
    main()
