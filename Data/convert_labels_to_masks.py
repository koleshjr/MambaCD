import os
import json
import numpy as np
from shapely.wkt import loads
from tqdm import tqdm
import cv2
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from concurrent.futures import ProcessPoolExecutor

# Argument parser setup
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--data", type=str, required=True, help="Path for saving preprocessed data")

ori_label_value_dict = {
    'background': (0, 0, 0),         # Black
    'no-damage': (255, 0, 0),       # Red
    'minor-damage': (0, 255, 0),    # Green
    'major-damage': (0, 0, 255),    # Blue
    'destroyed': (255, 255, 0)        # Yellow
}


def generate_and_save_image_with_masks(unique_label, labels_path, targets_path, image_size=(1024, 1024)):
    pre_path = os.path.join(labels_path, f"{unique_label}_pre_disaster.json")
    post_path = os.path.join(labels_path, f"{unique_label}_post_disaster.json")

    # Flags to track if post-disaster contains any classified features
    post_has_classified = False

    # Initialize a blank mask
    post_mask = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)  # RGB image mask
    pre_mask = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)  # RGB image mask

    for path in [pre_path, post_path]:
        # Load json file
        with open(path, 'r') as f:
            data = json.load(f)

        for feature in data['features']['xy']:
            try:
                poly = loads(feature["wkt"])

                # Attempt to buffer the polygon to fix validity issues
                if not poly.is_valid:
                    print(f"Invalid geometry detected: {feature['wkt']}. Attempting to buffer and fix.")
                    poly = poly.buffer(0)  # Try to fix invalid geometries by buffering (buffer distance is 0)
                    if not poly.is_valid:
                        print(f"Failed to fix geometry after buffering: {feature['wkt']}. Skipping.")
                        continue  # Skip if still invalid after buffering

                if 'post' in path:
                    # Get the damage type (subtype)
                    damage_type = feature["properties"].get("subtype", "background")
                    if damage_type == "un-classified":
                        continue  # Skip un-classified features for post-disaster

                    if damage_type not in ori_label_value_dict:
                        print(f"Unknown damage type: {damage_type}. Defaulting to background.")  # Default to background if unknown type
                        damage_type = 'background'  # Default to background if unknown type

                    post_has_classified = True  # Found a classified feature

                    color = ori_label_value_dict.get(damage_type, ori_label_value_dict["background"])

                    # Create an individual feature mask
                    feature_mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
                    int_coords = lambda x: np.array(x).round().astype(np.int32)
                    exteriors = [int_coords(poly.exterior.coords)]
                    cv2.fillPoly(feature_mask, exteriors, 255)  # Fill the polygon

                    # Now apply the color for the respective damage type to the mask
                    post_mask[feature_mask > 0] = color  # Apply color for the polygon

                elif 'pre' in path:
                    # For 'pre', mark the building presence with white and background as black
                    feature_mask = np.zeros(image_size, dtype=np.uint8)
                    int_coords = lambda x: np.array(x).round().astype(np.int32)
                    exteriors = [int_coords(poly.exterior.coords)]
                    cv2.fillPoly(feature_mask, exteriors, 1)

                    # Apply the white presence mask (1 for building presence, 0 for background)
                    pre_mask[feature_mask > 0] = 255  # Mark building with white color (255)

            except Exception as e:
                print(f"Error processing feature: {feature['wkt']}. Exception: {e}")
                continue

    # If there are no classified features in the post-disaster image, skip the entire pair
    if not post_has_classified:
        print(f"Skipping {unique_label} pair because the post-disaster image has no classified features.")
        return

    # Save both pre and post disaster masks as a valid pair
    output_pre_path = pre_path.replace(labels_path, targets_path).replace(".json", ".png")
    output_post_path = post_path.replace(labels_path, targets_path).replace(".json", ".png")

    # Convert post_mask to RGB before saving
    post_mask_rgb = cv2.cvtColor(post_mask, cv2.COLOR_BGR2RGB)
    pre_mask_rgb = cv2.cvtColor(pre_mask, cv2.COLOR_BGR2RGB)

    # Save the masks
    cv2.imwrite(output_pre_path, pre_mask_rgb)
    cv2.imwrite(output_post_path, post_mask_rgb)

def process_labels_in_parallel(labels_path, targets_path, unique_labels):
    # Use ProcessPoolExecutor to process each label in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        for unique_label in unique_labels:
            futures.append(executor.submit(generate_and_save_image_with_masks, unique_label, labels_path, targets_path))

        # Wait for all the tasks to complete
        for future in tqdm(futures):
            future.result()  # To catch exceptions raised by any task

if __name__ == "__main__":
    args = parser.parse_args()
    data_path = args.data
    labels_path = os.path.join(data_path, "labels")
    targets_path = os.path.join(data_path, "targets")
    os.makedirs(targets_path, exist_ok=True)

    all_labels = list(os.listdir(labels_path))
    unique_labels = list(set(["_".join(x.split("_")[:2]) for x in all_labels]))
    print(len(unique_labels), len(all_labels))

    # Process all unique labels in parallel
    process_labels_in_parallel(labels_path, targets_path, unique_labels)
