import cv2
from matplotlib import pyplot as plt
import numpy as np

def histogram_equalization_rgb(img_path, save_path):
    """
    Applies histogram equalization to each channel of an RGB image and saves the result.

    Args:
        img_path: Path to the input image file.
        save_path: Path to save the processed image.

    Returns:
        A numpy array representing the image with histogram equalized channels.
    """
    img = cv2.imread(img_path)

    # Split the image into its individual channels
    b, g, r = cv2.split(img)

    # Apply histogram equalization to each channel
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)

    # Merge the equalized channels back into a single image
    img_eq = cv2.merge([b_eq, g_eq, r_eq])

    # Save the equalized image
    cv2.imwrite(save_path, img_eq)

    return img_eq
