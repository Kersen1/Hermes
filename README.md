# Hermes

**Author:** Kersen

**Date:** September 2025  

Hermes is a Python-based assistant for scientific staff to facilitate image processing tasks.  
It provides a chat interface powered by a LLaMA-based language model and multiple image-processing tools that can be invoked on files or folders of images.

---

## Installation

```bash
pip install langchain_ollama langchain customtkinter matplotlib scikit-image scikit-learn
```
pull the Llama model 
```bash
ollama pull llama3.1:latest
```
##Main Tools

view_image_tool(path) – display images or folders of images.

contrast_changing_tool(path) – apply circular mask and logarithmic contrast.

equalization_tool(path) – logarithmic correction and histogram equalization.

histogram_tool(path) – show image with histogram.

fourier_tool(path) – 2D Fourier Transform visualization.

segmentation_tool(path) – Otsu thresholding and K-means clustering.

capabilities_tool(query) – lists all available tools.

exit_tool(query) – closes Hermes.


##FUNCTIONS THAT TOOLS USE

located in:
image_functions

show_histogram.show_image_with_histogram(path) – display image and histogram.

segment.image_segmentation(path) – Otsu + K-means segmentation.

fourier_analysis.fourier_texture_analysis(path) – Fourier transform visualization.

equalize.equalization(path) – contrast correction + histogram equalization.

contrast.process_contrast(path) – disk mask + logarithmic contrast.


#CHAT

ChatGUI

Textbox for chat history, entry field for input, send button.
