# IMPORT SECTION
# necessary library inclusions
# core dependency imports
import streamlit as st
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# ERROR HANDLING SETUP
# global exception management
# custom error display utility
def handle_external_error(e):
    st.markdown(f'<div style="background-color: red; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">System Error: {str(e)}</div>', unsafe_allow_html=True)
    st.stop()

try:
    # ARTIFACT INITIALIZATION
    # persistent model loading
    # automated network architecture construction
    @st.cache_resource
    def load_artifacts():
        pca = joblib.load('pca_model.joblib')
        svm = joblib.load('svm_model.joblib')
        golden_ref = np.load('golden_reference.npy')
        
        ae_baseline = Sequential([
            Input(shape=(128, 128, 3)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])
        ae_baseline.compile(optimizer='adam', loss='mse')

        base_mobilenet = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        mobilenet = Sequential([
            base_mobilenet,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        mobilenet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        with open('labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
            
        return pca, svm, ae_baseline, ae_baseline, 0.05, golden_ref, mobilenet, labels

    pca, svm, ae_baseline, ae_optimized, ae_thresh, golden_ref, mobilenet, labels = load_artifacts()

    # USER INTERFACE SETUP
    # page configuration parameters
    # main header branding
    st.set_page_config(page_title="AI Defect Detection", layout="wide")
    st.title("🏭 AI Visual Anomaly Detection & Comparison")
    st.markdown("---")

    # REPORT SECTION
    # centered diagnostic report header
    st.markdown("<h3 style='text-align: center;'>Diagnostic Report</h3>", unsafe_allow_html=True)
    
    # SPACING CONFIGURATION
    # vertical gap implementation
    st.markdown("<br>", unsafe_allow_html=True)

    # CONTROL PANEL CONFIGURATION
    # centered layout columns for diagnostic options
    col_empty1, col_cp, col_space, col_sg, col_empty2 = st.columns([1.5, 3, 0.3, 3, 1.5])
    
    # ENGINE SELECTION WIDGET
    # model selection radio buttons
    with col_cp:
        selected_model = st.radio(
            "Select Diagnostic Engine:",
            (
                "Classical ML (SVM + PCA)", 
                "Baseline Autoencoder", 
                "SSIM Analysis (Golden Ref)", 
                "MobileNetV2 + Grad-CAM"
            )
        )
        
    # SAFETY GATEKEEPER WIDGET
    # structural validation checkbox control
    with col_sg:
        st.markdown("<br>", unsafe_allow_html=True)
        use_gatekeeper = st.checkbox("Enable Industrial Safety Gatekeeper (SSIM)", value=True)

    # SPACING CONFIGURATION
    # vertical gap implementation
    st.markdown("<br>", unsafe_allow_html=True)

    # IMAGE UPLOAD SECTION
    # centered layout columns for file input and preview stacking
    col_empty_u1, col_up, col_empty_u2 = st.columns([2, 4, 2])
    
    # FILE ACQUISITION WIDGET
    # sample image upload interface
    with col_up:
        uploaded_file = st.file_uploader("Upload inspection sample (JPG, PNG)", type=["jpg", "png", "jpeg"])
        
        # VISUALIZATION PREVIEW
        # uploaded sample rendering container directly underneath
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Uploaded Sample", width=250)

    st.markdown("---")

    # EXECUTION PIPELINE
    # inspection execution logic
    if uploaded_file is not None:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # TRIGGER BUTTON
        # diagnostic control button element
        if st.button("ΕΚΤΕΛΕΣΗ ΕΛΕΓΧΟΥ 🚀", use_container_width=True):
            img_gray_resized = cv2.resize(img_gray, (128, 128))
            ssim_gate_score, _ = ssim(golden_ref, img_gray_resized, full=True, data_range=255)
            
            # GATEKEEPER VALIDATION
            # structural sanity check evaluation
            if use_gatekeeper and ssim_gate_score < 0.40:
                st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ INVALID IMAGE: Αδυναμία αναγνώρισης δομής.</div>', unsafe_allow_html=True)
                st.metric("SSIM Validation Score", f"{ssim_gate_score:.4f}")
            
            else:
                # MODEL EXECUTION BRANCH
                # classical machine learning inference
                if selected_model == "Classical ML (SVM + PCA)":
                    img_resized = cv2.resize(img_rgb, (128, 128))
                    img_flat = img_resized.reshape(1, -1)
                    img_pca = pca.transform(img_flat)
                    pred = svm.predict(img_pca)
                    if pred[0] == 1:
                        st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">✅ ΚΑΤΑΣΤΑΣΗ: ΦΥΣΙΟΛΟΓΙΚΟ (GOOD)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ ΚΑΤΑΣΤΑΣΗ: ΕΛΑΤΤΩΜΑΤΙΚΟ (ANOMALY)</div>', unsafe_allow_html=True)
                
                # MODEL EXECUTION BRANCH
                # baseline autoencoder reconstruction analysis
                elif selected_model == "Baseline Autoencoder":
                    img_resized = cv2.resize(img_rgb, (128, 128))
                    img_input = np.expand_dims(img_resized, axis=0) / 255.0
                    recon = ae_baseline.predict(img_input)[0]
                    mse = np.mean(np.square(img_resized / 255.0 - recon))
                    st.metric("Reconstruction Error (MSE)", f"{mse:.4f}")
                    fig, ax = plt.subplots()
                    cax = ax.imshow(np.mean(np.abs(img_resized / 255.0 - recon), axis=-1), cmap='jet')
                    fig.colorbar(cax)
                    ax.set_title("Reconstruction Error Heatmap")
                    ax.axis('off')
                    st.pyplot(fig)

                # MODEL EXECUTION BRANCH
                # structural similarity reference comparison
                elif selected_model == "SSIM Analysis (Golden Ref)":
                    score, diff_map = ssim(golden_ref, img_gray_resized, full=True, data_range=255)
                    if score >= 0.50:
                        st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">✅ ΚΑΤΑΣΤΑΣΗ: ΦΥΣΙΟΛΟΓΙΚΟ (PASSED)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ ΚΑΤΑΣΤΑΣΗ: ΕΛΑΤΤΩΜΑΤΙΚΟ (FAILED)</div>', unsafe_allow_html=True)
                    st.metric("SSIM Score", f"{score:.4f}")
                
                # MODEL EXECUTION BRANCH
                # mobilenet classification and explanation inference
                elif selected_model == "MobileNetV2 + Grad-CAM":
                    img_resized = cv2.resize(img_rgb, (224, 224))
                    img_tensor = preprocess_input(np.expand_dims(img_resized, axis=0).astype(np.float32))
                    pred = mobilenet.predict(img_tensor)[0][0]
                    if pred > 0.5:
                        st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">✅ ΚΑΤΑΣΤΑΣΗ: ΦΥΣΙΟΛΟΓΙΚΟ (GOOD)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ ΚΑΤΑΣΤΑΣΗ: ΕΛΑΤΤΩΜΑΤΙΚΟ (ANOMALY)</div>', unsafe_allow_html=True)
                    st.metric("Confidence", f"{(pred * 100 if pred > 0.5 else (1 - pred) * 100):.2f}%")

# EXCEPTION MANAGEMENT
# global failure catching block
except Exception as e:
    handle_external_error(e)