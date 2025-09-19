import matplotlib.pyplot as plt
from skimage import io, img_as_float
import os

def show_image_with_histogram(path: str):
    """
    Loads an image, draws an histogram and shows the image next to it.
    """
    if not os.path.exists(path):
        return f"File '{path}' does not exist."

    # Load image
    image = io.imread(path)
    image_float = img_as_float(image)

    # Create histogram
    if image_float.ndim == 3:  # colorful image
        # Show histogram for rgb
        colors = ('r', 'g', 'b')
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(image_float)
        ax[0].set_title("Original image")
        ax[0].axis("off")

        for i, color in enumerate(colors):
            ax[1].hist(image_float[..., i].ravel(), bins=256, color=color, alpha=0.6)
        ax[1].set_title("Histogram")
        ax[1].set_xlabel("Intensity")
        ax[1].set_ylabel("Number of pixels")
    else:  # grayscale image
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(image_float, cmap='gray')
        ax[0].set_title("Original image")
        ax[0].axis("off")

        ax[1].hist(image_float.ravel(), bins=256, color='black')
        ax[1].set_title("Histogram")
        ax[1].set_xlabel("Intensity")
        ax[1].set_ylabel("Number of pixels")

    plt.tight_layout()
    plt.show()

    return f"Processed image '{path}' successfully."