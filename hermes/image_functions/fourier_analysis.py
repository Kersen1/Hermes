import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os

def fourier_texture_analysis(path: str):
    """
    Loads an image, performs a 2D Fourier Transform, and visualizes
    the original image next to its magnitude spectrum.
    """
    if not os.path.exists(path):
        return f"File '{path}' does not exist."

    # Load the image in grayscale
    try:
        image = io.imread(path, as_gray=True)
    except Exception as e:
        return f"Error loading image: {e}"

    # Perform the 2D Fourier Transform
    f_transform = np.fft.fft2(image)

    # Shift the zero-frequency component to the center for better visualization
    # This places low-frequency components at the center and high-frequency ones at the edges.
    f_shift = np.fft.fftshift(f_transform)

    # Calculate the magnitude spectrum and apply a logarithmic scale.
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)

    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Display the magnitude spectrum
    ax[1].imshow(magnitude_spectrum, cmap='gray')
    ax[1].set_title("Fourier Transform Magnitude Spectrum")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

    return "Fourier transform analysis completed successfully."
