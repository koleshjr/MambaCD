import numpy as np
import cv2
from skimage.measure import label

ori_label_value_dict = {
    'background': (0, 0, 0),         # Black
    'no-damage': (255, 0, 0),       # Red
    'minor-damage': (0, 255, 0),    # Green
    'major-damage': (0, 0, 255),    # Blue
    'destroyed': (255, 255, 0)        # Yellow
}
def count_segments_in_mask(mask_image_path):
    # Load the mask image (either pre or post)
    mask = cv2.imread(mask_image_path, cv2.IMREAD_COLOR)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    label_type = 'post' if 'post' in mask_image_path else 'pre'
    
    # If the image is loaded successfully
    if mask is None:
        print(f"Error: Mask image at {mask_image_path} could not be loaded.")
        return None, None
    
    if label_type == 'post':
        # Dictionary to hold count of each class type
        class_counts = {key: 0 for key in ori_label_value_dict.keys()}
        # Label connected components (objects) in the binary mask
        grayscale_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(grayscale_mask, 1, 255, cv2.THRESH_BINARY)
        labeled_mask, num_labels = label(binary_mask, connectivity=2, return_num=True)

        # Iterate through each labeled segment (building)
        for label_id in range(1, num_labels + 1):
            # Extract each segment (building) mask
            building_mask = (labeled_mask == label_id).astype(np.uint8)

            # Get the majority color in the building segment
            segment_pixels = mask[building_mask == 1]
            unique_colors, counts = np.unique(segment_pixels, axis=0, return_counts=True)

            # Get the most frequent color
            majority_color = unique_colors[np.argmax(counts)]

            # Find the class that corresponds to the majority color
            for class_name, color in ori_label_value_dict.items():
                if np.array_equal(majority_color, color):
                    class_counts[class_name] += 1  # Increment the class count

        return labeled_mask, class_counts

    # For pre-disaster: simply count connected components (default behavior)
    else:
        # Convert to grayscale (assuming pre-disaster is binary)
        grayscale_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Ensure mask is binary (0 or 255 values)
        _, binary_mask = cv2.threshold(grayscale_mask, 1, 255, cv2.THRESH_BINARY)

        # Label connected components (objects) in the binary mask
        labeled_mask, num_labels = label(binary_mask, connectivity=2, return_num=True)

        return labeled_mask, num_labels

mask_image_path = 'test_images_labels_targets/targets/hurricane-florence_00000268_post_disaster.png'
_, num_labels = count_segments_in_mask(mask_image_path)
print(num_labels)

