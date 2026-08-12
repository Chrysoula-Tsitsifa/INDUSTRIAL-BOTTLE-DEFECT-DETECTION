# IMPORT SECTION
# Core dependencies for application environment.
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
# Custom dense layer for omission of quantization config parameter.
class SafeDense(Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

# ERROR HANDLING
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
        
        # OPTIMIZED AUTOENCODER
        # Full architecture and weights extraction from file without compilation metrics.
        ae_baseline = tf.keras.models.load_model('app/best_model_optimized.h5', compile=False)

        # MOBILENETV2
        # Full architecture and weights extraction with custom object scope.
        with tf.keras.utils.custom_object_scope({'Dense': SafeDense}):
            mobilenet = tf.keras.models.load_model('app/bottle_model_final_tuned.h5', compile=False)

        # CLASS LABELS
        # Reference classes from text file.
        with open('app/labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
            
        # ANOMALY THRESHOLD CALIBRATION
        # Presentation threshold for demonstration of global metric limitations.
        return pca, svm, ae_baseline, ae_baseline, 0.0015, golden_ref, mobilenet, labels

    pca, svm, ae_baseline, ae_optimized, ae_thresh, golden_ref, mobilenet, labels = load_artifacts()

    # USER INTERFACE
    # Strict absolute horizontal layout configuration.
    st.set_page_config(page_title="AI Defect Detection", layout="wide")

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

    # CONTROL PANEL
    # Balanced symmetrical layout columns for options with shifted checkboxes.
    col_cp, col_space, col_sg = st.columns([5.3, 0.2, 3.5])
    
    # STRUCTURAL INTEGRITY GATEKEEPER
    # Industrial safety mechanism for extreme anomaly blockage.
    with col_sg:
        use_gatekeeper = st.checkbox("Enable Industrial Safety Gatekeeper (SSIM)", value=True)
        
        # IMAGE ENHANCEMENT TOGGLE
        # Option for visual presentation enhancement via contrast filter.
        use_clahe = st.checkbox("Enable Mild CLAHE Filter", value=True)

    # ENGINE SELECTION
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

    # SPACING CONFIGURATION
    # Vertical gap implementation.
    st.markdown("<br>", unsafe_allow_html=True)

    # IMAGE UPLOAD
    # Perfectly centered layout columns for file input.
    col_empty_u1, col_up, col_empty_u2 = st.columns([2.5, 4, 2.5])
    
    # FILE ACQUISITION
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
                
                # CLASSICAL ML BRANCH
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
                
                # AUTOENCODER BRANCH
                # Baseline autoencoder reconstruction analysis.
                elif selected_model == "Baseline Autoencoder":
                    img_resized = cv2.resize(img_rgb, (128, 128))
                    img_input = np.expand_dims(img_resized, axis=0) / 255.0
                    recon = ae_baseline.predict(img_input)[0]
                    mse = np.mean(np.square(img_resized / 255.0 - recon))
                    
                    # ANOMALY THRESHOLD
                    # Evaluation of reconstruction error.
                    if mse > ae_thresh:
                        st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ STATUS: DEFECTIVE (ANOMALY)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">✅ STATUS: NORMAL (GOOD)</div>', unsafe_allow_html=True)
                    
                    st.metric("Reconstruction Error (MSE)", f"{mse:.4f}")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # RECONSTRUCTION HEATMAP
                    # Visual anomaly localization through mse map.
                    fig, ax = plt.subplots(figsize=(7, 5))
                    cax = ax.imshow(np.mean(np.abs(img_resized / 255.0 - recon), axis=-1), cmap='jet')
                    fig.colorbar(cax)
                    ax.set_title("Reconstruction Error Heatmap")
                    ax.axis('off')
                    
                    col_hm_e1, col_hm_mid, col_hm_e2 = st.columns([1.5, 7, 1.5])
                    with col_hm_mid:
                        st.pyplot(fig)

                # SSIM BRANCH
                # Structural similarity reference comparison.
                elif selected_model == "SSIM Analysis (Golden Ref)":
                    score, diff_map = ssim(golden_ref, img_gray_resized, full=True, data_range=255)
                    
                    # STRICT INDUSTRIAL THRESHOLD
                    # Evaluation limit for structural integrity.
                    if score >= 0.75:
                        st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">✅ STATUS: NORMAL (PASSED)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ STATUS: DEFECTIVE (FAILED)</div>', unsafe_allow_html=True)
                    st.metric("SSIM Score", f"{score:.4f}")
                    st.markdown("<br>", unsafe_allow_html=True)

                    # STRUCTURAL DIFFERENCE VISUALIZATION
                    # Anomaly emphasis via mathematical map inversion.
                    diff_visual = np.clip(1.0 - diff_map, 0, 1)
                    
                    fig_ssim, ax_ssim = plt.subplots(figsize=(7, 5))
                    cax_ssim = ax_ssim.imshow(diff_visual, cmap='jet', vmin=0, vmax=1)
                    fig_ssim.colorbar(cax_ssim)
                    ax_ssim.set_title("SSIM Difference Map (Anomalies Highlighted)")
                    ax_ssim.axis('off')
                    
                    col_ssim_e1, col_ssim_mid, col_ssim_e2 = st.columns([1.5, 7, 1.5])
                    with col_ssim_mid:
                        st.pyplot(fig_ssim)
                
                # MOBILENETV2 BRANCH
                # Mobilenet classification and explanation inference.
                elif selected_model == "MobileNetV2 + Grad-CAM":

                    # IMAGE PREPROCESSING
                    # Format preparation for neural network.
                    img_for_inference = img_rgb.copy()
                    
                    # SOFT CLAHE ENHANCEMENT
                    # Subtle contrast adaptation for structure clarity without noise amplification.
                    if use_clahe:
                        lab = cv2.cvtColor(img_for_inference, cv2.COLOR_RGB2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
                        l = clahe.apply(l)
                        img_for_inference = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

                    # ROI CROPPING
                    # Manual alignment for bottleneck orifice focus.
                    h, w = img_for_inference.shape[:2]
                    center_y, center_x = h // 2, w // 2
                    size = 200 
                    img_resized = cv2.resize(img_for_inference, (224, 224))
                    
                    # PREPROCESSING FIX
                    # Application of official mobilenetv2 scaling limits on standard inputs.
                    img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
                    img_tensor = preprocess_input(img_array)
                    
                    # ROBUST INFERENCE
                    # Neural evaluation on preprocessed data stream.
                    pred = mobilenet.predict(img_tensor)[0][0]
                    
                    # UNCERTAINTY ZONE
                    # Boundaries for human audit requirement.
                    if 0.40 <= pred <= 0.60:
                        st.markdown('<div style="background-color: orange; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">⚠️ UNCERTAINTY: HUMAN AUDIT REQUIRED</div>', unsafe_allow_html=True)
                    
                    # OPTIMAL THRESHOLD
                    # Mathematical boundary for final decision.
                    elif pred > 0.60:
                        st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">✅ STATUS: NORMAL (GOOD)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ STATUS: DEFECTIVE (ANOMALY)</div>', unsafe_allow_html=True)
                    st.metric("Confidence", f"{(pred * 100 if pred > 0.5 else (1 - pred) * 100):.2f}%")
                    st.markdown("<br>", unsafe_allow_html=True)

                    # GRAD CAM LOCALIZATION
                    # Visual anomaly regions via decoupled neural architecture.
                    base_model = mobilenet.layers[0]
                    
                    last_conv_layer = None
                    for layer in reversed(base_model.layers):
                        if len(layer.output.shape) == 4:
                            last_conv_layer = layer
                            break
                    
                    if last_conv_layer:
                        conv_model = tf.keras.models.Model(inputs=base_model.inputs, outputs=last_conv_layer.output)
                        
                        # CLASSIFIER ISOLATION
                        # Standalone diagnostic head for backpropagation mapping.
                        classifier_input = tf.keras.Input(shape=conv_model.output.shape[1:])
                        x = classifier_input
                        for layer in mobilenet.layers[1:]:
                            x = layer(x)
                        classifier_model = tf.keras.models.Model(inputs=classifier_input, outputs=x)
                        
                        # GRADIENT CALCULATION
                        # Saliency mapping via differentiable forward and backward passes.
                        with tf.GradientTape() as tape:
                            last_conv_layer_output = conv_model(img_tensor)
                            tape.watch(last_conv_layer_output)
                            
                            preds = classifier_model(last_conv_layer_output)
                            
                            # CLASS CHANNEL SELECTION
                            # Dimension evaluation for specific node gradient.
                            if preds.shape[-1] == 1:
                                
                                # BINARY GRADIENT INVERSION
                                # Inversion of probability channel for defective class.
                                if pred <= 0.60:
                                    class_channel = 1.0 - preds[:, 0]
                                else:
                                    class_channel = preds[:, 0]
                            else:
                                class_idx = tf.argmax(preds[0])
                                class_channel = preds[:, class_idx]
                        
                        grads = tape.gradient(class_channel, last_conv_layer_output)
                        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                        
                        # HEATMAP SYNTHESIS
                        # Matrix multiplication and extraction as numpy array.
                        heatmap_tensor = tf.reduce_sum(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)[0]
                        heatmap = np.maximum(heatmap_tensor.numpy(), 0)
                        
                        if np.max(heatmap) != 0:
                            heatmap /= np.max(heatmap)
                        else:
                            heatmap = heatmap * 0.0
                            
                        heatmap_resized = cv2.resize(heatmap, (224, 224))
                    else:
                        heatmap_resized = np.zeros((224, 224))

                    # VISUAL OVERLAY
                    # Coloration and fusion with designated background matrix.
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    superimposed = np.clip(heatmap_colored * 0.4 + img_resized, 0, 255).astype(np.uint8)
                    
                    # DIAGNOSTIC PLOT
                    # Matplotlib synthesis of final heat overlay.
                    fig_cam, ax_cam = plt.subplots(figsize=(7, 5))
                    ax_cam.imshow(superimposed)
                    ax_cam.set_title("Grad-CAM Heatmap Overlay")
                    ax_cam.axis('off')
                    
                    col_cam_e1, col_cam_mid, col_cam_e2 = st.columns([1.5, 7, 1.5])
                    with col_cam_mid:
                        st.pyplot(fig_cam)

except Exception as e:
    handle_external_error(e)