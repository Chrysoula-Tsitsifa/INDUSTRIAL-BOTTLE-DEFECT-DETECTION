# IMPORT SECTION
# Core dependencies.
import streamlit as st
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# COMPATIBILITY LAYER
# Custom dense layer for the omission of the quantization config parameter.
class SafeDense(Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

# ERROR HANDLING SETUP
# Global exception management.
def handle_external_error(e):
    st.markdown(f'<div style="background-color: red; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">System Error: {str(e)}</div>', unsafe_allow_html=True)
    st.stop()

try:
    # ARTIFACT INITIALIZATION
    # Persistent models and reference data from application folder.
    @st.cache_resource
    def load_artifacts():
        pca = joblib.load('app/pca_model.joblib')
        svm = joblib.load('app/svm_model.joblib')
        golden_ref = np.load('app/golden_reference.npy')
        
        # OPTIMIZED AUTOENCODER LOADING
        # Full architecture and weights extraction from file without compilation metrics.
        ae_baseline = tf.keras.models.load_model('app/best_model_optimized.h5', compile=False)

        # MOBILENETV2 LOADING
        # Full architecture and weights extraction with custom object scope.
        with tf.keras.utils.custom_object_scope({'Dense': SafeDense}):
            mobilenet = tf.keras.models.load_model('app/bottle_model_final_tuned.h5', compile=False)

        # CLASS LABELS EXTRACTION
        # Reference classes from text file.
        with open('app/labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
            
        return pca, svm, ae_baseline, ae_baseline, 0.05, golden_ref, mobilenet, labels

    pca, svm, ae_baseline, ae_optimized, ae_thresh, golden_ref, mobilenet, labels = load_artifacts()

    # USER INTERFACE SETUP
    # Page configuration parameters.
    st.set_page_config(page_title="AI Defect Detection", layout="wide")

    # CSS CENTERING INJECTION
    # Strict absolute horizontal layout configuration.
    st.markdown("""
        <style>
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            margin: auto;
        }
        .centered-label {
            text-align: center;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>🏭 AI Visual Anomaly Detection & Comparison</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # REPORT SECTION
    # Centered diagnostic report header.
    st.markdown("<h3 style='text-align: center;'>Diagnostic Report</h3>", unsafe_allow_html=True)
    
    # SPACING CONFIGURATION
    # Vertical gap implementation.
    st.markdown("<br>", unsafe_allow_html=True)

    # CONTROL PANEL CONFIGURATION
    # Balanced symmetrical layout columns for options with gatekeeper shift to the far right edge.
    col_cp, col_space, col_sg = st.columns([5.3, 0.2, 3.5])
    
    # ENGINE SELECTION WIDGET
    # Model selection radio buttons.
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
        
    # STRUCTURAL INTEGRITY GATEKEEPER
    # Industrial safety mechanism for extreme anomaly blockage.
    with col_sg:
        st.markdown("<br><br>", unsafe_allow_html=True)
        use_gatekeeper = st.checkbox("Enable Industrial Safety Gatekeeper (SSIM)", value=True)

    # SPACING CONFIGURATION
    # Vertical gap implementation.
    st.markdown("<br>", unsafe_allow_html=True)

    # IMAGE UPLOAD SECTION
    # Perfectly centered layout columns for file input.
    col_empty_u1, col_up, col_empty_u2 = st.columns([2.5, 4, 2.5])
    
    # FILE ACQUISITION WIDGET
    # Sample image upload interface with centered label.
    with col_up:
        st.markdown("<div class='centered-label'>Upload inspection sample (JPG, PNG)</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    # VISUALIZATION PREVIEW
    # Perfectly aligned preview container match for input columns.
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        col_img_e1, col_img_mid, col_img_e2 = st.columns([2.5, 4, 2.5])
        with col_img_mid:
            st.image(img_rgb, caption="Uploaded Sample", use_container_width=True)

    st.markdown("---")

    # EXECUTION PIPELINE
    # Inspection logic execution.
    if uploaded_file is not None:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # TRIGGER BUTTON
        # Diagnostic control button element.
        if st.button("RUN INSPECTION 🚀", use_container_width=True):
            img_gray_resized = cv2.resize(img_gray, (128, 128))
            ssim_gate_score, _ = ssim(golden_ref, img_gray_resized, full=True, data_range=255)
            
            # GATEKEEPER VALIDATION
            # Structural sanity check evaluation.
            if use_gatekeeper and ssim_gate_score < 0.40:
                st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ INVALID IMAGE: Structural recognition failure.</div>', unsafe_allow_html=True)
                st.metric("SSIM Validation Score", f"{ssim_gate_score:.4f}")
            
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                
                # MODEL EXECUTION BRANCH
                # Classical machine learning inference.
                if selected_model == "Classical ML (SVM + PCA)":
                    img_resized = cv2.resize(img_rgb, (128, 128))
                    img_flat = img_resized.reshape(1, -1)
                    img_pca = pca.transform(img_flat)
                    pred = svm.predict(img_pca)
                    if pred[0] == 1:
                        st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">✅ STATUS: NORMAL (GOOD)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ STATUS: DEFECTIVE (ANOMALY)</div>', unsafe_allow_html=True)
                
# MODEL EXECUTION BRANCH
                # Baseline autoencoder reconstruction analysis.
                elif selected_model == "Baseline Autoencoder":
                    img_resized = cv2.resize(img_rgb, (128, 128))
                    img_input = np.expand_dims(img_resized, axis=0) / 255.0
                    recon = ae_baseline.predict(img_input)[0]
                    mse = np.mean(np.square(img_resized / 255.0 - recon))
                    
                    # ANOMALY DETECTION THRESHOLD
                    # Evaluation of the reconstruction error.
                    if mse > ae_thresh:
                        st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ STATUS: DEFECTIVE (ANOMALY)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">✅ STATUS: NORMAL (GOOD)</div>', unsafe_allow_html=True)
                    
                    st.metric("Reconstruction Error (MSE)", f"{mse:.4f}")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # RECONSTRUCTION HEATMAP GENERATION
                    # Visual anomaly localization through mse map.
                    fig, ax = plt.subplots(figsize=(7, 5))
                    cax = ax.imshow(np.mean(np.abs(img_resized / 255.0 - recon), axis=-1), cmap='jet')
                    fig.colorbar(cax)
                    ax.set_title("Reconstruction Error Heatmap")
                    ax.axis('off')
                    
                    col_hm_e1, col_hm_mid, col_hm_e2 = st.columns([1.5, 7, 1.5])
                    with col_hm_mid:
                        st.pyplot(fig)

                # MODEL EXECUTION BRANCH
                # Structural similarity reference comparison.
                elif selected_model == "SSIM Analysis (Golden Ref)":
                    score, diff_map = ssim(golden_ref, img_gray_resized, full=True, data_range=255)
                    if score >= 0.50:
                        st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">✅ STATUS: NORMAL (PASSED)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ STATUS: DEFECTIVE (FAILED)</div>', unsafe_allow_html=True)
                    st.metric("SSIM Score", f"{score:.4f}")
                    st.markdown("<br>", unsafe_allow_html=True)

                    # STRUCTURAL DIFFERENCE VISUALIZATION
                    # Spatial disparity map for ssim analysis.
                    fig_ssim, ax_ssim = plt.subplots(figsize=(7, 5))
                    cax_ssim = ax_ssim.imshow(diff_map, cmap='jet')
                    fig_ssim.colorbar(cax_ssim)
                    ax_ssim.set_title("SSIM Difference Map")
                    ax_ssim.axis('off')
                    
                    col_ssim_e1, col_ssim_mid, col_ssim_e2 = st.columns([1.5, 7, 1.5])
                    with col_ssim_mid:
                        st.pyplot(fig_ssim)
                
                # MODEL EXECUTION BRANCH
                # Mobilenet classification and explanation inference.
                elif selected_model == "MobileNetV2 + Grad-CAM":

		    # ROI CROPPING
		    # Manual centering to force focus on the bottle orifice.
		    h, w = img_rgb.shape[:2]
		    center_y, center_x = h // 2, w // 2
		    size = 200

		    img_rgb = img_rgb[center_y-size:center_y+size, center_x-size:center_x+size]

                    img_resized = cv2.resize(img_rgb, (224, 224))
                    img_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
                    pred = mobilenet.predict(img_tensor)[0][0]
                    
                    # UNCERTAINTY ZONE
                    # Boundaries for the human audit requirement.
                    if 0.40 <= pred <= 0.60:
                        st.markdown('<div style="background-color: orange; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">⚠️ UNCERTAINTY: HUMAN AUDIT REQUIRED</div>', unsafe_allow_html=True)
                    
                    # OPTIMAL THRESHOLD
                    # Mathematical boundary for the final decision.
                    elif pred > 0.60:
                        st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">✅ STATUS: NORMAL (GOOD)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ STATUS: DEFECTIVE (ANOMALY)</div>', unsafe_allow_html=True)
                    st.metric("Confidence", f"{(pred * 100 if pred > 0.5 else (1 - pred) * 100):.2f}%")
                    st.markdown("<br>", unsafe_allow_html=True)

                    # GRAD CAM LOCALIZATION
                    # Visual anomaly regions via gradient flows.
                    with tf.GradientTape() as tape:
                        inputs = tf.cast(img_tensor, tf.float32)
                        conv_outputs = mobilenet.layers[0](inputs)
                        tape.watch(conv_outputs)
                        x = mobilenet.layers[1](conv_outputs)
                        x = mobilenet.layers[2](x)
                        x = mobilenet.layers[3](x)
                        predictions = mobilenet.layers[4](x)
                    
                    grads = tape.gradient(predictions, conv_outputs)
                    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)[0]
                    heatmap = np.maximum(heatmap, 0)
                    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1e-10
                    
                    heatmap_resized = cv2.resize(heatmap, (224, 224))
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    superimposed = np.clip(heatmap_colored * 0.4 + img_resized, 0, 255).astype(np.uint8)
                    
                    fig_cam, ax_cam = plt.subplots(figsize=(7, 5))
                    ax_cam.imshow(superimposed)
                    ax_cam.set_title("Grad-CAM Heatmap Overlay")
                    ax_cam.axis('off')
                    
                    col_cam_e1, col_cam_mid, col_cam_e2 = st.columns([1.5, 7, 1.5])
                    with col_cam_mid:
                        st.pyplot(fig_cam)

except Exception as e:
    handle_external_error(e)