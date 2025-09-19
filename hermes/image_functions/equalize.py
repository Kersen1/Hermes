import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_float, exposure
import os


def equalization(path: str):
    """
    Gives image an logarithmic correction and equalizes the histogram.
    """
    if not os.path.exists(path):
        return f"File '{path}' does not exist."
    
    # Load image
    image = io.imread(path)
    image_float = img_as_float(image)

    # Logarithmic correction
    log_image = exposure.adjust_log(image_float, 0.65)

    # Histogram equalization  
    equalized_image = exposure.equalize_hist(image_float)

    # Show image
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image_float)
    ax[0].set_title("Oryginalny obraz")
    ax[0].axis("off")

    ax[1].imshow(log_image)
    ax[1].set_title("Logarytmiczna korekcja")
    ax[1].axis("off")

    ax[2].imshow(equalized_image)
    ax[2].set_title("Histogram equalization ")
    ax[2].axis("off")

    plt.show()
    
    return f"Processed image '{path}' successfully."