# 🏭 INDUSTRIAL BOTTLE DEFECT DETECTION

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://industrial-bottle-defect-detection-nhartvdccuurmpvggaquo9.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-GHCR-2496ED?style=flat&logo=docker&logoColor=white)](https://github.com/Chrysoula-Tsitsifa/INDUSTRIAL-BOTTLE-DEFECT-DETECTION/pkgs/container/industrial-bottle-defect-detection)

Production-oriented industrial computer-vision project for bottle defect detection. The repository covers the full R&D path from classical anomaly-detection baselines and autoencoders to a calibrated MobileNetV2 classifier, explainability, artifact integrity checks, automated tests, Streamlit deployment, and a reproducible Docker image.

<br>

## 🚀 LIVE DEMO

**Open the deployed application:**

👉 [Industrial Bottle Defect Detection — Streamlit](https://industrial-bottle-defect-detection-nhartvdccuurmpvggaquo9.streamlit.app/)

The live app supports image upload, multi-model inference, calibrated decision reporting, and Grad-CAM visualization for the MobileNetV2 deployment candidate.

<br>

## 🔍 WHAT THE APPLICATION DEMONSTRATES

- **Classical anomaly-detection baseline:** PCA-based dimensionality reduction followed by One-Class SVM benchmarking.
- **Convolutional autoencoder baseline:** Reconstruction-based anomaly detection using a simple deep-learning reference model.
- **Controlled autoencoder optimization:** Tuned convolutional autoencoder training with regularization, Early Stopping, learning-rate reduction, and threshold selection.
- **Structural reconstruction analysis:** SSIM-based image similarity analysis, reconstruction diagnostics, and sensitivity testing.
- **Transfer learning and calibrated classification:** MobileNetV2 fine-tuning, probability calibration, operational thresholding, and final Good/Defective decision logic.
- **Explainable AI:** Grad-CAM generation, spatial evidence visualization, and locked XAI evaluation workflows.
- **Deployment engineering:** Streamlit application delivery, Docker containerization, GHCR publication, artifact integrity checks, and automated tests.

<br>

## 🧠 DEPLOYMENT CANDIDATE

**The Streamlit application defaults to the calibrated MobileNetV2 classifier. It reports:**

- Calibrated `P(Good)`.
- Operational decision threshold.
- Predicted class.
- Raw neural probability before calibration.
- Optional Grad-CAM evidence.

The application is a **host-side deployment candidate**. The exported TFLite model is included for edge deployment work, while independent target-device benchmarking remains a separate validation step.

<br>

## 🐳 DOCKER

**A public container image is automatically built and published to GitHub Container Registry.**

For a reviewer or employer, these commands reproduce the application locally without manually configuring Python or installing project dependencies.

Pull the published image:

```bash
docker pull ghcr.io/chrysoula-tsitsifa/industrial-bottle-defect-detection:latest
```

Run the container and expose the Streamlit port:

```bash
docker run --rm -p 8501:8501 ghcr.io/chrysoula-tsitsifa/industrial-bottle-defect-detection:latest
```

Open the application in a browser:

```text
http://localhost:8501
```

The image is built from the repository `Dockerfile` and published through GitHub Actions. The public Streamlit demo can also be used directly without Docker.

<br>

## 🛠️ TECHNOLOGY STACK

- **Language / Runtime:** Python 3.12.
- **Deep Learning / Edge AI:** TensorFlow 2.21, Keras 3.14, TensorFlow Lite.
- **Machine Learning / Calibration:** scikit-learn 1.6.1, PCA, One-Class SVM, probability calibration, and operational thresholding.
- **Computer Vision / Structural Analysis:** Pillow, scikit-image 0.26, and SSIM.
- **Explainability:** Grad-CAM.
- **Numerical Computing / Visualization:** NumPy 2.4 and Matplotlib 3.10.
- **Model Serialization / Artifacts:** Joblib, Keras model artifacts, JSON metric contracts, and SHA-256 manifests.
- **Application / Deployment:** Streamlit 1.61, Docker, and GitHub Container Registry (GHCR).
- **Automation / Reproducibility:** GitHub Actions, Docker Buildx, Dev Containers, automated tests, and artifact-integrity validation.
- **R&D Workflow:** Jupyter Notebooks, Git, and GitHub.

<br>

## 📂 REPOSITORY STRUCTURE

```text
INDUSTRIAL-BOTTLE-DEFECT-DETECTION/                         # Project root.
├── .devcontainer/                                         # Reproducible VS Code development environment.
│   └── devcontainer.json                                  # Dev Container configuration.
├── .github/                                               # GitHub automation configuration.
│   └── workflows/                                         # GitHub Actions workflows.
│       └── docker.yml                                     # Builds and publishes the Docker image to GHCR.
├── app/                                                   # Production Streamlit application package.
│   ├── __init__.py                                        # Marks the application directory as a Python package.
│   ├── main.py                                            # Streamlit application entrypoint.
│   ├── core/                                              # Core inference and validation logic.
│   │   ├── __init__.py                                    # Marks the core directory as a Python package.
│   │   ├── artifacts.py                                   # Resolves and validates frozen deployment artifacts.
│   │   ├── contracts.py                                   # Defines deployment and model contracts.
│   │   ├── inference.py                                   # Loads models and executes multi-model inference.
│   │   ├── preprocessing.py                               # Decodes and prepares input images.
│   │   ├── validation.py                                  # Performs inference-input validation.
│   │   └── xai.py                                         # Generates and validates Grad-CAM evidence.
│   └── ui/                                                # Presentation layer for the Streamlit interface.
│       ├── __init__.py                                    # Marks the UI directory as a Python package.
│       ├── components.py                                  # Reusable Streamlit UI components.
│       └── styles.py                                      # Application styling and layout rules.
├── demo images/                                           # Public bottle samples for quick application testing.
├── evaluation/                                            # Locked XAI and localization evaluation workflows.
│   ├── results/                                           # Saved evaluation outputs and benchmark results.
│   ├── xai_localization_benchmark.py                      # Quantitative Grad-CAM localization benchmark.
│   └── xai_locked_test_evaluation.py                      # Locked-test XAI evaluation pipeline.
├── final_artifacts/                                       # Canonical frozen models, metrics, and integrity metadata.
│   ├── artifact_manifest.json                             # Artifact inventory and SHA-256 integrity manifest.
│   ├── baseline_metrics.json                              # Baseline benchmark metrics.
│   ├── notebook1_artifacts/                               # Frozen artifacts from the baseline-model stage.
│   ├── notebook2_artifacts/                               # Frozen artifacts from the tuned-autoencoder stage.
│   ├── notebook2_metrics.json                             # Tuned-autoencoder evaluation metrics.
│   ├── notebook3_artifacts/                               # Frozen artifacts from the SSIM-analysis stage.
│   ├── notebook3_metrics.json                             # SSIM structural-analysis metrics.
│   ├── notebook4_artifacts/                               # Frozen artifacts from the transfer-learning stage.
│   └── notebook4_metrics.json                             # Transfer-learning and deployment metrics.
├── notebooks/                                             # Four-stage research and experimentation pipeline.
│   ├── 01_baseline_models_and_benchmarking.ipynb          # Classical and deep-learning baseline benchmarking.
│   ├── 02_convolutional_autoencoder_controlled_tuning.ipynb # Controlled convolutional-autoencoder optimization.
│   ├── 03_ssim_structural_reconstruction_analysis.ipynb   # SSIM-based structural reconstruction analysis.
│   └── 04_transfer_learning_xai_and_deployment_evaluation.ipynb # Transfer learning, XAI, and deployment evaluation.
├── tests/                                                 # Automated application and deployment tests.
│   ├── test_deployment_integrity.py                       # Validates frozen deployment artifacts and integrity.
│   ├── test_inference.py                                  # Tests model loading and inference paths.
│   ├── test_preprocessing.py                              # Tests image decoding and preprocessing.
│   ├── test_validation.py                                 # Tests inference-input validation rules.
│   └── test_xai.py                                        # Tests Grad-CAM generation and XAI safeguards.
├── .dockerignore                                          # Excludes unnecessary files from Docker build context.
├── .gitattributes                                         # Defines Git attribute behavior.
├── .gitignore                                             # Excludes local, generated, and temporary files from Git.
├── Dockerfile                                             # Defines the production container image.
├── LICENSE                                                # MIT License.
├── requirements.txt                                       # Pinned Python runtime dependencies.
└── README.md                                              # Project overview and deployment documentation.
```

The tree intentionally expands the important source, evaluation, test, and deployment files while summarizing bulk artifact and demo-image contents so the README remains readable.

<br>

## 📓 RESEARCH PIPELINE

### PHASE 1 — CLASSICAL BASELINES & BENCHMARKING

- A deterministic preprocessing and benchmarking path establishes a reproducible reference point for later deep-learning models.
- PCA is used for dimensionality reduction before One-Class SVM anomaly detection, providing a classical machine-learning control baseline.
- A simple convolutional autoencoder provides the first reconstruction-based deep-learning baseline for direct comparison against the classical approach.
- Baseline outputs and metrics are frozen into deployment artifacts so later improvements can be measured against a stable reference.

### PHASE 2 — CONTROLLED CONVOLUTIONAL AUTOENCODER TUNING

- The convolutional autoencoder is expanded into a controlled optimization workflow with parameterized training choices and regularization.
- Early Stopping and learning-rate reduction are used to stabilize training and reduce unnecessary overfitting during tuning.
- Reconstruction-error distributions are analyzed to support threshold selection and class-separability assessment rather than relying on an arbitrary cutoff.
- The selected model, metrics, and supporting artifacts are exported for downstream structural analysis and deployment comparison.

### PHASE 3 — SSIM STRUCTURAL RECONSTRUCTION ANALYSIS

- Structural Similarity Index (SSIM) is used to evaluate reconstruction quality beyond pixel-level error alone.
- Structural degradation and reconstruction behavior are analyzed to determine how image-quality changes affect anomaly-detection reliability.
- Sensitivity testing examines the response of the pipeline under controlled perturbations and supports evidence-based quality thresholds.
- The resulting metrics and artifacts provide an additional structural-quality layer for later forensic and deployment analysis.

### PHASE 4 — TRANSFER LEARNING, CALIBRATION, XAI & DEPLOYMENT

- MobileNetV2 transfer learning is used to build the supervised Good/Defective deployment candidate, followed by controlled fine-tuning and generalization checks.
- Raw neural outputs are converted into calibrated probabilities and combined with an operational decision threshold for reproducible classification decisions.
- Error-analysis workflows combine prediction confidence with structural evidence to support forensic inspection of difficult or misclassified samples.
- Grad-CAM provides post-hoc spatial evidence, while dedicated locked-test and localization benchmarks evaluate XAI behavior independently of the live UI.
- Latency and deployment-oriented evaluation support practical runtime assessment, and the finalized network is exported to TensorFlow Lite for edge deployment work.
- Deployment artifacts, contracts, integrity manifests, Streamlit delivery, Docker packaging, and automated validation complete the transition from research pipeline to reproducible application.

<br>

## ✅ REPRODUCIBILITY & VALIDATION

**The deployment package includes:**

- Canonical frozen model artifacts.
- Model and metric contracts.
- Artifact inventory with SHA-256 verification.
- Automated preprocessing, input-validation, inference, deployment-integrity, and XAI tests.
- Dedicated locked XAI evaluation and localization benchmark outputs.
- Pinned runtime dependencies for reproducible application execution.
- GitHub Actions automation for Docker build and GHCR publication.
- Dev Container configuration for a consistent development environment.

<br>

## 📄 LICENSE

MIT License.
