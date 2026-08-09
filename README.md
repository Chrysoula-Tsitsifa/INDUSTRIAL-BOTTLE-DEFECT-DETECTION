# 🏭 Industrial Visual Anomaly Detection & Edge AI Pipeline

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](ΤΟ_ΔΙΚΟ_ΣΟΥ_LINK_ΕΔΩ)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)

An enterprise grade, end to end Machine Learning pipeline engineered for automated visual inspection and structural anomaly detection in manufacturing lines. This project demonstrates a complete MLOps lifecycle - From classical algorithmic benchmarking and deep latent space optimization, to Explainable AI (XAI) diagnostics, edge IoT quantization, and system telemetry.

<br>

## 🚀 Live Interactive Deployment
Test the production ready Streamlit application equipped with a global Out of Distribution (OOD) Safety Gatekeeper:  
👉 **[Launch Streamlit Application](YOUR_STREAMLIT_LIVE_LINK_HERE)**

<br>

## 🧠 Core Architecture & R&D Evolution (The 4 Phase Pipeline)

The system was developed through a rigorous 4 phase research and engineering pipeline, ensuring mathematical justification for every architectural decision.
<br>
### Phase 1: Algorithmic Benchmarking & Baselines
* **Classical ML Control Group:** Engineered an idempotent data ingestion pipeline and applied Principal Component Analysis (PCA) for dimensionality reduction, followed by a One Class SVM to establish a strict statistical baseline.
* **Deep Learning Baseline:** Constructed a foundational Convolutional Autoencoder (CAE) for unsupervised anomaly detection.
* **Defect Localization:** Translated Mean Squared Error (MSE) reconstruction loss into normalized spatial error heatmaps for interpretable visual detection.

### Phase 2: Architectural Engineering & Regularization
* **Deep Latent Space Optimization:** Designed a parameterized, modular Convolutional Autoencoder incorporating Batch Normalization to mitigate internal covariate shift and stabilize the training of complex structural anomalies.
* **Automated Hyperparameter Tuning:** Engineered a robust Grid Search pipeline governed by dynamic learning rate scheduling (`ReduceLROnPlateau`) and Early Stopping.
* **Class Separability Diagnostics:** Established production ready thresholding by analyzing bimodal error distributions, visually proving iterative performance gains (ROC AUC) against the Classical ML baselines.

### Phase 3: Image Quality Assurance (IQC) & Data Integrity
* **Structural Degradation Profiling:** Engineered an automated Image Quality Control (IQC) pipeline utilizing OpenCV and the Structural Similarity Index (SSIM).
* **Sensitivity Analysis & Thresholding:** Executed comprehensive multi value stress tests simulating factory environments (Gaussian noise, luminance shifts) to mathematically quantify the exact "Breaking Point" of structural integrity before downstream Deep Learning tasks.
* **Automated Reporting:** Developed a deterministic Pandas consolidation module to generate structured CSV sensitivity reports.

### Phase 4: Transfer Learning, XAI, Forensics & Edge Deployment
* **Dataset Restructuring:** Implemented Stratified Splitting to transition the dataset from unsupervised to supervised formats.
* **Generalization Stability:** Engineered a Two Phase Transfer Learning architecture using MobileNetV2. Conducted rigorous generalization gap analysis, proving that targeted fine tuning eliminated initial overfitting and yielded a highly stable model (improving Validation Loss from 0.1085 to 0.1073).
* **Forensic Error Analysis (Wall of Shame):** Developed an automated Industrial Viability Report that categorizes misclassifications into distinct diagnostic Zones by mathematically correlating Model Confidence with SSIM.
* **Explainable AI (XAI):** Integrated CLAHE enhanced Grad CAM with sub pixel centroid localization to detect and mitigate **Shortcut Learning** (the *Clever Hans Effect*), proving robustness against spurious environmental correlations (e.g., glare).
* **Manifold Analytics & Latency:** Validated latent space separability using PCA initialized t SNE manifold mapping (Version 8) and Silhouette scoring. Conducted speed tests for latency profiling.
* **Edge AI & Quantization:** Exported the finalized neural architecture into a highly compressed TensorFlow Lite (TFLite) binary optimized for IoT edge deployment.
* **Telemetry Profiling:** Executed a comprehensive System Resource Audit to monitor footprint constraints.

<br>

## 🔬 Advanced Statistical Diagnostics
Beyond standard accuracy, this pipeline is evaluated using stringent industrial and statistical metrics:
* **Wilson Confidence Interval**
* **Brier Score (Calibration)**
* **Expected Calibration Error (ECE)**
* **Matthews Correlation Coefficient (MCC)**

<br>

## 🛠️ Technology Stack
* **Deep Learning & Edge AI:** TensorFlow, Keras, TFLite
* **Machine Learning:** Scikit Learn (PCA, One Class SVM, Grid Search)
* **Computer Vision & Image Processing:** OpenCV, Scikit Image (SSIM)
* **Web UI & MLOps:** Streamlit
* **Data Engineering & Visualization:** Pandas, NumPy, Matplotlib, Seaborn

 <br>  

## 📂 Repository Structure

```text
Industrial Bottle Defect Detection/
│
├── app/                                              # Production UI & API
│   ├── app.py                                        # Streamlit Application
│   ├── requirements.txt                              # Dependency Management
│   └── *.keras / *.joblib / *.npy                    # Serialized Models & Golden References
│
├── notebooks/                                        # The 4 Phase R&D Pipeline
│   ├── 01_baseline_and_benchmarking.ipynb
│   ├── 02_deep_learning_optimization.ipynb
│   ├── 03_data_augmentation_quality_control.ipynb
│   └── 04_transfer_learning_xai_and_edge.ipynb
│
└── README.md                                         # Technical Documentation
