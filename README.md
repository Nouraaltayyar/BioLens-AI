# AgroMind: BioLens AI 

## Overview
This repository contains the complete machine learning pipeline for **AgroMind**, a robust plant disease classification system developed as a capstone project. It handles automated dataset balancing, benchmarks multiple deep learning architectures, and deploys the highest-performing model via a Streamlit web interface called **BioLens AI**. 

To ensure system integrity during live inference, the application utilizes an LLM-powered "GenAI Gatekeeper" to validate user uploads before passing them to the PyTorch classifier.

## Key Features
* **Automated Data Balancing:** Dynamically oversamples and augments the PlantVillage dataset so that all 38 distinct crop and disease classes are perfectly balanced at exactly 2,200 images each.
* **Architecture Benchmarking:** Trains and evaluates three specific convolutional neural networks, utilizing early stopping and continuous validation tracking to crown an automatic winner.
* **GenAI Gatekeeper:** Integrates GPT-4o-mini (via OpenRouter) to pre-screen user uploads in real-time. It uses strict prompt engineering to verify if the uploaded file is actually a plant leaf before allowing the AI to process it.
* **Dynamic Streamlit Deployment:** Automatically transitions the best-performing model's state dictionary (`.pth`) into active production memory and serves the interactive BioLens AI web app using an Ngrok tunnel.

## Model Performance Comparison
The automated benchmarking script tracked training across 15 epochs for each model. Below are the final evaluation metrics:

| Architecture | Peak Validation Accuracy | Best Validation Loss |
| :--- | :--- | :--- |
| **EFFICIENTNET_B0** | **0.9949** | **0.0041** |
| **RESNET18** | 0.9924 | 0.0068 |
| **MOBILENET_V2** | 0.9889 | 0.0113 |

*Result: The pipeline automatically designated EfficientNet-B0 as the winner and loaded it into the GenAI Gatekeeper's memory space.*

## Code Structure: Inference & Deployment
The deployment phase seamlessly connects the computer vision model with the web front-end:

* **The Gatekeeper (`validate_leaf`)**: Converts the user's PIL image to base64 and queries the OpenAI API. It requires a strict `YES` or `NO` response to the prompt: *"Is this a plant leaf or any part of a plant?"*
* **Class Mapping**: Extracts the 38 labeled directories into a sorted array and saves them to `class_names.json` to ensure the model's output neurons correctly map to the human-readable disease names.
* **Streamlit Tunneling**: Executes `biolens_app.py` in a headless server state, utilizing `pyngrok` to expose the local port 8501 to the public web for immediate user access.

---

## Getting Started

### Prerequisites
Ensure you have Python installed along with the necessary deep learning and web frameworks. You can install the core dependencies via `pip`:

```bash
# Core Machine Learning & Data Processing
pip install torch torchvision albumentations mlflow pandas numpy opencv-python tqdm

# Web App & LLM Integration
pip install streamlit pyngrok openai
