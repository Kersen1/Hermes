import matplotlib.pyplot as plt
from skimage import io, filters, exposure
from sklearn.cluster import MiniBatchKMeans
import os

def image_segmentation(path: str):
    """
    Performs image segmentation using Otsu's thresholding and k-means clustering.
    """
    if not os.path.exists(path):
        return f"File '{path}' does not exist."

    # Load the image in grayscale and convert to float
    try:
        image = io.imread(path, as_gray=True)
        image = exposure.rescale_intensity(image, out_range=(0, 1))
    except Exception as e:
        return f"Error loading image: {e}"

    # --- 1. Otsu's Thresholding ---
    try:
        threshold = filters.threshold_otsu(image)
        binary_otsu = image > threshold
    except ValueError:
        return "Could not perform Otsu's thresholding. The image might have too few distinct intensity levels."

    # --- 2. K-means Clustering ---
    # Reshape the image to a 2D array of pixels and their intensity
    rows, cols = image.shape
    image_reshaped = image.reshape(-1, 1)

    # Initialize and train the K-means model with k=2 clusters
    # We choose k=2 to mimic the binary segmentation of Otsu's method for comparison.
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, n_init=3)
    kmeans.fit(image_reshaped)
    
    # Get the labels for each pixel
    labels = kmeans.labels_
    
    # Reshape the labels back into an image
    segmented_kmeans = labels.reshape(rows, cols)

    # --- 3. Visualization ---
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Original Image
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Otsu's Segmentation
    ax[1].imshow(binary_otsu, cmap='gray')
    ax[1].set_title("Otsu's Thresholding")
    ax[1].axis("off")

    # K-means Segmentation
    ax[2].imshow(segmented_kmeans, cmap='gray')
    ax[2].set_title("K-means Clustering (k=2)")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

    return "Image segmentation completed successfully."
