import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching_parabolic_rgb(img_path, save_path, power=0.5):
    """
    Stretches the contrast of an RGB image using a parabolic function (y = x^power)
    while preserving the RGB color channels.

    Args:
        img_path: Path to the input image file.
        save_path: Path to save the processed image.
        power: The power value for the parabolic function (default is 0.5).

    Returns:
        The contrast-stretched RGB image as a NumPy array.
    """
    img = cv2.imread(img_path)

    # Apply parabolic contrast stretching to each color channel separately
    image_stretched = np.zeros_like(img)
    for channel in range(3):
        image_stretched[:, :, channel] = (img[:, :, channel].astype(np.float32) / 255.0) ** power * 255.0
        image_stretched[:, :, channel] = image_stretched[:, :, channel].astype(np.uint8)

    # Save the contrast-stretched image
    cv2.imwrite(save_path, image_stretched)

    return image_stretched
