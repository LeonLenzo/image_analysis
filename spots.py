import os
from PIL import Image
import numpy as np
import cv2
from skimage import morphology
import pandas as pd

def normalize_hue(h):
    # Normalize hue from [0, 360] to [0, 179]
    return int(h * 179 / 255)

# Function to process a single image
def process_image(image_path, healthy_hue, chlorosis_hue, necrosis_hue, output_directory):
    image = Image.open(image_path)
    image_np = np.array(image)

    # Convert image to HSV
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # Normalize the hue values to range [0, 179]
    healthy_hue = normalize_hue(healthy_hue)
    chlorosis_hue = normalize_hue(chlorosis_hue)
    necrosis_hue = normalize_hue(necrosis_hue)

    # Create binary masks
    background_mask = cv2.inRange(hsv_image, np.array([healthy_hue, 0, 0], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    leaf_mask = cv2.inRange(hsv_image, np.array([0, 0, 0], dtype=np.uint8), np.array([healthy_hue, 255, 255], dtype=np.uint8))
    healthy_mask = cv2.inRange(hsv_image, np.array([chlorosis_hue, 0, 0], dtype=np.uint8), np.array([healthy_hue, 255, 255], dtype=np.uint8))
    chlorosis_mask = cv2.inRange(hsv_image, np.array([necrosis_hue, 0, 0], dtype=np.uint8), np.array([chlorosis_hue, 255, 255], dtype=np.uint8))
    necrosis_mask = cv2.inRange(hsv_image, np.array([0, 0, 0], dtype=np.uint8), np.array([necrosis_hue, 255, 255], dtype=np.uint8))

    # Clean leaf and background masks
    leaf_mask_cleaned = morphology.remove_small_objects(leaf_mask.astype(bool), 500)
    leaf_mask_cleaned = morphology.remove_small_holes(leaf_mask_cleaned, 500)
    background_mask_cleaned = morphology.remove_small_objects(background_mask.astype(bool), 500)
    background_mask_cleaned = morphology.remove_small_holes(background_mask_cleaned, 500)

    # Clean chlorosis and necrosis masks
    chlorosis_mask_cleaned = morphology.remove_small_objects(chlorosis_mask.astype(bool), 300)
    chlorosis_mask_cleaned = morphology.remove_small_holes(chlorosis_mask_cleaned, 300)
    necrosis_mask_cleaned = morphology.remove_small_objects(necrosis_mask.astype(bool), 300)
    necrosis_mask_cleaned = morphology.remove_small_holes(necrosis_mask_cleaned, 300)

    # Calculate area
    leaf_area = np.sum(leaf_mask_cleaned)
    healthy_area = np.sum(healthy_mask != 0)
    chlorosis_area = np.sum(chlorosis_mask_cleaned != 0)
    necrosis_area = np.sum(necrosis_mask_cleaned != 0)

    # Calculate percentages
    healthy_percentage = (healthy_area / leaf_area) * 100 if leaf_area > 0 else 0
    chlorosis_percentage = (chlorosis_area / leaf_area) * 100 if leaf_area > 0 else 0
    necrosis_percentage = (necrosis_area / leaf_area) * 100 if leaf_area > 0 else 0

    # Create the images to save
    leaf_image_no_background = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
    leaf_image_no_background[background_mask_cleaned != 0] = [0, 0, 0, 0]  # Remove background with transparency

    # Create simplified image with solid masks
    mask_image = leaf_image_no_background.copy()
    mask_image[chlorosis_mask_cleaned != 0] = [255, 255, 0, 128]  # Bright Yellow for chlorosis
    mask_image[necrosis_mask_cleaned != 0] = [255, 153, 0, 200]  # Brown for necrosis

    # Blend the original image with the mask image
    blended_image = cv2.addWeighted(leaf_image_no_background, 1, mask_image, 0.5, 0)

    # Save the images
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    leaf_image_path = os.path.join(output_directory, f"{base_filename}_isolated.png")
    simplified_image_path = os.path.join(output_directory, f"{base_filename}_masked.png")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    Image.fromarray(leaf_image_no_background).save(leaf_image_path)
    Image.fromarray(blended_image).save(simplified_image_path)

    return {
        "filename": base_filename,
        "healthy_area": healthy_area,
        "chlorosis_area": chlorosis_area,
        "necrosis_area": necrosis_area,
        "leaf_area": leaf_area,
        "healthy_percentage": healthy_percentage,
        "chlorosis_percentage": chlorosis_percentage,
        "necrosis_percentage": necrosis_percentage
    }

# Function to process all images in a directory
def process_directory(directory_path, healthy_hue, chlorosis_hue, necrosis_hue):
    output_directory = os.path.join(directory_path, "output")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    results = []
    for filename in os.listdir(directory_path):
        if filename.endswith((".tif", ".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory_path, filename)
            result = process_image(image_path, healthy_hue, chlorosis_hue, necrosis_hue, output_directory)
            results.append(result)
    return results

# Prompt user for inputs
directory_path = input("Please enter the path to the image directory: ")
healthy_hue = 120  # Fixed example hue for healthy
chlorosis_hue = int(input("Enter the hue value for the chlorosis threshold: "))
necrosis_hue = int(input("Enter the hue value for the necrosis threshold: "))

# Process all images in the directory
print("Starting processing...")
results = process_directory(directory_path, healthy_hue, chlorosis_hue, necrosis_hue)
print("Processing complete.")

# Convert results to a DataFrame and save to CSV
df = pd.DataFrame(results)
output_csv_path = os.path.join(directory_path, "output", "summary_results.csv")
df.to_csv(output_csv_path, index=False)

# Print the DataFrame
print(df)
