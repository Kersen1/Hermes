import matplotlib.pyplot as plt
import numpy as np
from skimage import io, draw, exposure
import os

def process_contrast(path: str):
    """
    Applies disk mask and logarithmic correction to an image and displays it.
    """
    if not os.path.exists(path):
        return f"File '{path}' does not exist."
    
    image = io.imread(path)
    h, w = image.shape[:2]
    center = (h // 2, w // 2)  # (y, x)

    # Create circular mask
    mask = np.zeros((h, w), dtype=bool)
    rr, cc = draw.disk(center, 600, shape=(h, w))
    mask[rr, cc] = True

    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]

    # Logarithmic correction
    logarithmic_corrected = exposure.adjust_log(masked_image, 0.65)

    # Display results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(masked_image)
    ax[1].set_title("Masked Image")
    ax[1].axis("off")

    ax[2].imshow(logarithmic_corrected)
    ax[2].set_title("Logarithmic Correction on Masked")
    ax[2].axis("off")

    plt.show()
    
    return f"Processed image '{path}' successfully."
