# AI for Sustainable Development: Poverty Prediction from Satellite Imagery

> **Coursework 2 Submission** | UCL
> **Student:** [Your Name/Student ID]
> **Module:** AI for Sustainable Development

## 📌 Project Overview
This project is a reproduction and extension of the seminal paper *"Combining satellite imagery and machine learning to predict poverty"* (Jean et al., 2016).

The goal is to predict economic well-being using high-resolution satellite imagery. This repository contains:
1.  **Replication:** A faithful reproduction of the baseline methodology using DHS data from 5 African countries (Nigeria, Tanzania, Uganda, Malawi, Rwanda).
2.  **Adaptation (Methodology):** Upgrading the CNN backbone from the original **VGG-F** to a deeper, residual-based **ResNet-18** to improve feature extraction capabilities.
3.  **Adaptation (Context):** Applying the trained model to a new geographic context: **Guizhou, China**, to analyze its ability to distinguish between "Policy-driven Poverty Alleviation" (e.g., relocation sites) and "Organic Economic Growth".

## 🛠️ Key Features
* **ResNet-18 Implementation:** A custom TensorFlow 1.15 implementation of ResNet-18 (`models/models_resnet.py`).
* **China Dataset:** A curated dataset of 20 locations in Guizhou, China, including adversarial samples for robustness testing (`china_coordinates.csv`).
* **Explainability:** Saliency map generation to visualize model attention (`visualize_saliency.py`).

### 🔍 Model Explainability Preview
An example of Saliency Map generation on Guizhou satellite imagery (Untrained Baseline):
![Saliency Map](images/saliency_test.png)

## 💻 Installation & Setup

This project uses **TensorFlow 1.15** and **Python 3.7**.

```bash
# 1. Clone the repository
git clone https://github.com/Quarkgluonmixture/africa_china_poverty.git
cd africa_china_poverty

# 2. Create Conda Environment
conda env create -f env.yml
conda activate poverty_env

# 3. Authenticate Google Earth Engine
earthengine authenticate
```

## 🚀 How to Run

### 1. Test Model Architecture
Verify that the ResNet-18 model is built correctly:

```bash
python test_resnet18.py
```

### 2. Run Inference on China Data
Test the forward pass on the curated Chinese satellite images:

```bash
python test_china_inference_flow.py
```

### 3. Generate Saliency Maps
Visualize what the model "sees":

```bash
python visualize_saliency.py
```

---
*Note: This repository is for academic assessment purposes.*
