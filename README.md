# 🏭 Industrial Bottle Defect Detection

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://industrial-bottle-defect-detection-nhartvdccuurmpvggaquo9.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-GHCR-2496ED?style=flat&logo=docker&logoColor=white)](https://github.com/Chrysoula-Tsitsifa/INDUSTRIAL-BOTTLE-DEFECT-DETECTION/pkgs/container/industrial-bottle-defect-detection)

Production-oriented industrial computer-vision project for bottle defect detection. The repository covers the full R&D path from classical anomaly-detection baselines and autoencoders to a calibrated MobileNetV2 classifier, explainability, artifact integrity checks, automated tests, Streamlit deployment, and a reproducible Docker image.



## 🚀 Live Demo

**Open the deployed application:**  

👉 [Industrial Bottle Defect Detection — Streamlit](https://industrial-bottle-defect-detection-nhartvdccuurmpvggaquo9.streamlit.app/)

The live app supports image upload, multi-model inference, calibrated decision reporting, and Grad-CAM visualization for the MobileNetV2 deployment candidate.


## 🔍 What the application demonstrates

- **Notebook 1 — Classical baseline:** PCA + One-Class SVM and a simple autoencoder.
- **Notebook 2 — Tuned autoencoder:** controlled convolutional-autoencoder optimization and threshold selection.
- **Notebook 3 — Structural analysis:** SSIM-based reconstruction diagnostics and sensitivity analysis.
- **Notebook 4 — Deployment candidate:** MobileNetV2 transfer learning, probability calibration, operational thresholding, Grad-CAM explainability, latency analysis, and TFLite export.
- **Validated deployment artifacts:** frozen models, metrics, contracts, SHA-256 integrity manifest, and deployment checks.
- **Automated testing:** preprocessing, input validation, inference, XAI, and deployment-integrity tests.


## 🧠 Deployment Candidate

The Streamlit application defaults to the calibrated MobileNetV2 classifier. It reports:

- calibrated `P(Good)`
- operational decision threshold
- predicted class
- raw neural probability before calibration
- optional Grad-CAM evidence

The application is a **host-side deployment candidate**. The exported TFLite model is included for edge deployment work, while independent target-device benchmarking remains a separate validation step.


## 🐳 Docker

A public container image is automatically built and published to GitHub Container Registry.

Pull the image:

```bash
docker pull ghcr.io/chrysoula-tsitsifa/industrial-bottle-defect-detection:latest
```

Run it:

```bash
docker run --rm -p 8501:8501 ghcr.io/chrysoula-tsitsifa/industrial-bottle-defect-detection:latest
```

Then open:

```text
http://localhost:8501
```

The image is built from the repository `Dockerfile` and published through GitHub Actions.


## 🛠️ Technology Stack

- **Deep Learning / Edge AI:** TensorFlow, Keras, TFLite
- **Machine Learning:** scikit-learn, PCA, One-Class SVM
- **Computer Vision:** Pillow, scikit-image, SSIM
- **Explainability:** Grad-CAM
- **Application:** Streamlit
- **Data / Evaluation:** NumPy, pandas, Matplotlib
- **MLOps / Reproducibility:** Docker, GitHub Actions, GHCR, artifact manifests, automated tests


## 📂 Repository Structure

```text
INDUSTRIAL-BOTTLE-DEFECT-DETECTION/
├── app/
│   ├── main.py                  # Streamlit entrypoint
│   ├── core/                    # inference, preprocessing, validation, XAI
│   └── ui/                      # UI components and styles
├── demo images/                 # sample bottle images for quick testing
├── evaluation/                  # locked XAI / localization evaluation scripts and results
├── final_artifacts/             # frozen models, metrics, contracts, integrity manifest
├── notebooks/                   # four-stage R&D pipeline
├── tests/                       # deployment and inference test suite
├── .github/workflows/           # automated Docker build and publish workflow
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```


## 📓 Research Pipeline

### Phase 1 — Baselines
Classical PCA + One-Class SVM and a simple convolutional autoencoder establish reference performance.

### Phase 2 — Controlled Autoencoder Tuning
A tuned convolutional autoencoder is evaluated under controlled optimization and threshold-selection procedures.

### Phase 3 — Structural Similarity Analysis
SSIM is used to assess structural reconstruction quality and sensitivity to image degradation.

### Phase 4 — Transfer Learning, Calibration, XAI & Edge Export
MobileNetV2 transfer learning is followed by probability calibration, operational decision logic, error analysis, Grad-CAM explainability, latency profiling, and TFLite export.


## ✅ Reproducibility & Validation

The deployment package includes:

- canonical frozen artifacts
- model/metric contracts
- artifact inventory with SHA-256 verification
- automated preprocessing and inference tests
- dedicated XAI tests and locked evaluation reports
- GitHub Actions Docker build and GHCR publication

## 📄 License

MIT License.
