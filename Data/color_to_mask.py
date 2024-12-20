from PIL import Image
import numpy as np
import imageio


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img

# Load the image
image_path = "test_images_labels_targets/targets/hurricane-florence_00000268_post_disaster.png"
clf_label = img_loader(image_path)#np.array(Image.open(image_path))

# Check the shape of the image
print("Shape of clf_label:", clf_label.shape)

# Check the unique pixel values in the image
unique_colors = np.unique(clf_label.reshape(-1, clf_label.shape[-1]), axis=0)
print("Unique colors in the image:", unique_colors)

def convert_color_to_class( clf_label):
    """
    Converts a color-coded label image to class indices dynamically.
    This assumes that unique colors in the image are predefined.
    """
    color_to_class = {
        (0, 0, 0): 0,         # Background
        (255, 0, 0): 1,       # No damage (Red)
        (0, 255, 0): 2,       # Minor damage (Green)
        (0, 0, 255): 3,       # Major damage (Blue)
        (255, 255, 0): 4      # Destroyed (Yellow)
    }


    # Convert to indices
    h, w, c = clf_label.shape
    clf_label_indices = np.zeros((h, w), dtype=np.uint8)

    # Find unique colors and map them to indices
    for color, class_idx in color_to_class.items():
        mask = (clf_label[:, :, 0] == color[0]) & (clf_label[:, :, 1] == color[1]) & (clf_label[:, :, 2] == color[2])
        clf_label_indices[mask] = class_idx

    return clf_label_indices

# Convert the label
clf_label_converted = convert_color_to_class(clf_label)
# Check unique classes in the converted label
unique_classes, counts = np.unique(clf_label_converted, return_counts=True)
print("Unique classes and their counts in clf_label_converted:")
for cls, count in zip(unique_classes, counts):
    print(f"Class {cls}: {count} pixels")