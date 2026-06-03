<div align="center">

# 🔭 HERMES
### AI-Powered Image Analysis Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-agent-green?logo=chainlink)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.1%20%7C%20llava-black?logo=ollama)](https://ollama.com)
[![scikit-image](https://img.shields.io/badge/scikit--image-CV-orange)](https://scikit-image.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A desktop chatbot that lets scientific staff analyse images using plain English commands — powered by a local LLM agent.*

</div>

---

## ✨ What is HERMES?

HERMES is a desktop chat application where you type natural language commands and an LLM agent automatically selects and runs the right image processing tool. No clicking through menus — just describe what you want.

```
You:    describe C:\scans\sample.jpg
HERMES: The image shows a microscopic cross-section with distinct
        layered regions. The upper region appears lighter in tone...

You:    segment C:\scans\sample.jpg
HERMES: [opens matplotlib window with Otsu + K-means segmentation]
```

---

## 🛠️ Features

| Tool | Description |
|------|-------------|
| 👁️ **View** | Display any image or entire folder |
| 🤖 **AI Vision** | LLaVA multimodal LLM describes image contents in natural language |
| 🎛️ **Contrast** | Circular disk mask + logarithmic correction |
| 📊 **Histogram** | Side-by-side image and RGB/grayscale histogram |
| ⚖️ **Equalization** | Logarithmic correction and histogram equalization |
| 〰️ **Fourier** | 2D Fourier Transform for texture and frequency analysis |
| ✂️ **Segmentation** | Otsu thresholding and K-means clustering (k=2) |

All tools support both **single files** and **entire folders**.

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Agent | LangChain + LangGraph `create_react_agent` |
| Language model | Ollama `llama3.1:latest` (local, offline) |
| Vision model | Ollama `llava:latest` (multimodal, local) |
| Image processing | `scikit-image`, `numpy` |
| Segmentation | `scikit-learn` MiniBatchKMeans |
| Visualisation | `matplotlib` |
| GUI | `customtkinter` (dark mode) |

---

## 🚀 Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/hermes.git
cd hermes
```

**2. Install Python dependencies**
```bash
pip install langchain langchain-ollama langgraph customtkinter matplotlib scikit-image scikit-learn
```

**3. Install [Ollama](https://ollama.com) and pull the models**
```bash
ollama pull llama3.1:latest   # ~5 GB — main language model
ollama pull llava:latest      # ~4 GB — vision model for AI image description
```

**4. Run**
```bash
python main.py
```

> ⚠️ Ollama must be running in the background. On first install it starts automatically; if not, run `ollama serve` in a separate terminal.

---

## 📁 Project Structure

```
hermes/
├── main.py                  # Agent, all tools, and GUI
├── image_functions/
│   ├── contrast.py          # Disk mask + logarithmic contrast correction
│   ├── equalize.py          # Histogram equalization
│   ├── show_histogram.py    # Histogram visualisation
│   ├── fourier_analysis.py  # 2D Fourier Transform
│   └── segment.py           # Otsu thresholding + K-means segmentation
├── images/                  # Sample images
└── README.md
```

---

## 💬 Example Commands

```
help
show C:\images\scan.jpg
describe C:\images\scan.jpg
apply contrast to C:\images\
show histogram of C:\images\sample.png
run fourier analysis on C:\images\texture.jpg
segment C:\images\brain.png
exit
```

---

## 🔮 Roadmap

- [ ] Edge detection tool (Canny, Sobel)
- [ ] Noise reduction (Gaussian, median filter)
- [ ] PDF report generation from analysis results
- [ ] Support for OpenAI GPT-4o vision
- [ ] Batch processing progress bar

---

## 👤 Author

**Kersen1** — built September 2025, updated June 2026

---

<div align="center">
<i>Built with LangChain · scikit-image · Ollama · customtkinter</i>
</div>

ChatGUI

Textbox for chat history, entry field for input, send button.
