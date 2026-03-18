# 🫀 HeartSPECT Segmentation Project

This repository contains the source code, trained models, and an interactive web prototype for **Left Ventricle Segmentation** on Cardiac SPECT Imaging using deep learning (3D U-Net).

This project was built to demonstrate **clinical applicability, safety, interpretability, and reproducibility** in automating myocardial perfusion scintigraphy (MPS) analysis.

---

## 🚀 Quick Start: Web Prototype

A ready-to-use local web prototype is provided to demonstrate the model's capabilities in real-time. It features a multi-plane 3D viewer, automatic inference, and a probability map (confidence overlay).

1. **Install Dependencies**
   ```bash
   pip install -r prototype/requirements.txt
   ```
   *(For GPU acceleration, install the appropriate CUDA PyTorch version from [pytorch.org](https://pytorch.org/).)*

2. **Run the Prototype**
   ```bash
   streamlit run prototype/app.py
   ```
   The app will automatically open in your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
heartSPECT/
├── models/             ← Trained PyTorch checkpoints (e.g., best_model.pth)
├── prototype/          ← Streamlit web app UI & inference pipeline
│   ├── app.py          ← Main Streamlit application
│   ├── model.py        ← UNet3D architecture definition
│   └── utils.py        ← Preprocessing and utility functions
├── notebooks/          ← Jupyter notebooks for EDA, training, and evaluation
├── scripts/            ← Utility scripts (e.g., metadata extraction)
├── outputs/            ← Visualizations, training curves, and evaluation plots
├── docs/               ← Dataset guidelines and licensing
└── data/               ← [IGNORED BY GIT] Raw DICOM and processed NIfTI data
```

> **Note on Data:** Due to medical data privacy guidelines and repository size limits, the raw DICOM files and preprocessed NIfTI datasets are **not included** in this GitHub repository. The `.gitignore` rule automatically safely excludes them. 
> 
> However, we have included **5 sample DICOM files** in `prototype/sample_data/` so the web app can be tested out-of-the-box. The trained model weights (~67MB) are also included.
>
> 📥 **Full Dataset Download:** To acquire the complete dataset used in this project, please download the ZIP file directly from [PhysioNet: Myocardial Perfusion SPECT (1.0.0)](https://physionet.org/content/myocardial-perfusion-spect/get-zip/1.0.0/) and extract the DICOM files into the `data/raw/DICOM/` directory.

---

## 🧠 Methodology

### 1. Data Preprocessing
- Volumetric resampling to a uniform `(64, 64, 64)` structure (Target Shape).
- Percentile-based intensity clipping (1st - 99th percentile) to handle anomalies.
- Z-score normalization for robust model input.

### 2. Model Architecture
- **3D U-Net** implemented in PyTorch.
- **Encoder:** 16, 32, 64, 128 channels.
- **Bottleneck:** 256 channels.
- **Output:** Sigmoid activation mapping to a probability mask.
- Total Parameters: **~5.6 Million**.

### 3. Loss & Training
- Mixed objective: `0.5 * Binary Cross Entropy (BCE) + 0.5 * Dice Loss` to handle the severe class imbalance (left ventricle is typically ~1% of the entire volume).
- **Optimizer:** AdamW with Cosine Annealing Learning Rate scheduling.

---

## 📊 Performance & Evaluation
Our evaluation is measured on a held-out test set (unseen data) using the following key clinical metrics:
- **Dice Similarity Coefficient (DSC)**
- **Intersection over Union (IoU)**
- **Hausdorff Distance (HD95)**

*Detailed evaluation reports and visualizations can be found in the `outputs/` directory and the main Jupyter notebook.*

---
## ⚖️ Disclaimer

⚠️ **Academic & Research Use Only:** This tool is a research prototype. It is **NOT** intended for clinical diagnosis. Always consult a qualified medical professional for final diagnostic decisions.
