import streamlit as st
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# GLOBAL ERROR HANDLER
def handle_external_error(e):
    st.markdown(f'<div style="background-color: red; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">System Error: {str(e)}</div>', unsafe_allow_html=True)
    st.stop()

try:
    # ARTIFACT LOADING
    @st.cache_resource
    def load_artifacts():
        # Διόρθωση import για το Functional API στο TF 2.15
        from tensorflow.keras.models import Model as Functional
        custom_objects = {
            'quantization_config': None,
            'Functional': Functional
        }
        
        with tf.keras.utils.custom_object_scope(custom_objects):
            pca = joblib.load('pca_model.joblib')
            svm = joblib.load('svm_model.joblib')
            ae_baseline = load_model('baseline_ae.keras')
            ae_optimized = load_model('optimized_ae.keras')
            ae_thresh = joblib.load('optimized_ae_threshold.joblib')
            golden_ref = np.load('golden_reference.npy')
            mobilenet = load_model('mobilenet_v2_final.keras')
            
        with open('labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return pca, svm, ae_baseline, ae_optimized, ae_thresh, golden_ref, mobilenet, labels

    pca, svm, ae_baseline, ae_optimized, ae_thresh, golden_ref, mobilenet, labels = load_artifacts()

    # GRAD-CAM ALGORITHM
    def make_gradcam_heatmap(img_array, full_model, last_conv_layer_name='out_relu'):
        base_model = full_model.layers[0]
        target_layer = base_model.get_layer(last_conv_layer_name)
        conv_model = tf.keras.Model(inputs=base_model.inputs, outputs=target_layer.output)
        
        classifier_input = tf.keras.Input(shape=conv_model.output.shape[1:])
        x = classifier_input
        for layer in full_model.layers[1:]: 
            x = layer(x)
        classifier_model = tf.keras.Model(inputs=classifier_input, outputs=x)
        
        with tf.GradientTape() as tape:
            last_conv_layer_output = conv_model(img_array)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            class_idx = tf.argmax(preds[0])
            class_channel = preds[:, class_idx]
            
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(last_conv_layer_output[0] * pooled_grads, axis=-1)
        return (tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)).numpy()

    # UI INITIALIZATION
    st.set_page_config(page_title="AI Defect Detection", layout="wide")
    st.title("🏭 AI Visual Anomaly Detection & Comparison")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Control Panel")
        selected_model = st.radio(
            "Select Diagnostic Engine:",
            (
                "Classical ML (SVM + PCA)", 
                "Baseline Autoencoder", 
                "Optimized Autoencoder", 
                "SSIM Analysis (Golden Ref)", 
                "MobileNetV2 + Grad-CAM"
            )
        )
        
        st.markdown("---")
        use_gatekeeper = st.checkbox("Enable Industrial Safety Gatekeeper (SSIM)", value=True)
        uploaded_file = st.file_uploader("Upload inspection sample (JPG, PNG)", type=["jpg", "png", "jpeg"])

    with col2:
        st.subheader("Diagnostic Report")
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            st.image(img_rgb, caption="Uploaded Sample", width=300)
            
            if st.button("ΕΚΤΕΛΕΣΗ ΕΛΕΓΧΟΥ 🚀", use_container_width=True):
                img_gray_resized = cv2.resize(img_gray, (128, 128))
                ssim_gate_score, _ = ssim(golden_ref, img_gray_resized, full=True, data_range=255)
                
                if use_gatekeeper and ssim_gate_score < 0.40:
                    st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ INVALID IMAGE: Αδυναμία αναγνώρισης δομής.</div>', unsafe_allow_html=True)
                    st.metric("SSIM Validation Score", f"{ssim_gate_score:.4f}")
                
                else:
                    if selected_model == "Classical ML (SVM + PCA)":
                        img_resized = cv2.resize(img_rgb, (128, 128))
                        img_flat = img_resized.reshape(1, -1)
                        img_pca = pca.transform(img_flat)
                        pred = svm.predict(img_pca)
                        if pred[0] == 1:
                            st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">✅ ΚΑΤΑΣΤΑΣΗ: ΦΥΣΙΟΛΟΓΙΚΟ (GOOD)</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ ΚΑΤΑΣΤΑΣΗ: ΕΛΑΤΤΩΜΑΤΙΚΟ (ANOMALY)</div>', unsafe_allow_html=True)
                    
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

                    elif selected_model == "Optimized Autoencoder":
                        img_resized = cv2.resize(img_rgb, (128, 128))
                        img_input = np.expand_dims(img_resized, axis=0) / 255.0
                        recon = ae_optimized.predict(img_input)[0]
                        mse = np.mean(np.square(img_resized / 255.0 - recon))
                        if mse <= ae_thresh:
                            st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">✅ ΚΑΤΑΣΤΑΣΗ: ΦΥΣΙΟΛΟΓΙΚΟ (GOOD)</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ ΚΑΤΑΣΤΑΣΗ: ΕΛΑΤΤΩΜΑΤΙΚΟ (ANOMALY)</div>', unsafe_allow_html=True)
                        col_a, col_b = st.columns(2)
                        col_a.metric("MSE", f"{mse:.4f}")
                        col_b.metric("Threshold", f"{ae_thresh:.4f}")
                        
                    elif selected_model == "SSIM Analysis (Golden Ref)":
                        score, diff_map = ssim(golden_ref, img_gray_resized, full=True, data_range=255)
                        if score >= 0.50:
                            st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">✅ ΚΑΤΑΣΤΑΣΗ: ΦΥΣΙΟΛΟΓΙΚΟ (PASSED)</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div style="background-color: red; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ ΚΑΤΑΣΤΑΣΗ: ΕΛΑΤΤΩΜΑΤΙΚΟ (FAILED)</div>', unsafe_allow_html=True)
                        st.metric("SSIM Score", f"{score:.4f}")
                    
                    elif selected_model == "MobileNetV2 + Grad-CAM":
                        img_resized = cv2.resize(img_rgb, (224, 224))
                        img_tensor = preprocess_input(np.expand_dims(img_resized, axis=0).astype(np.float32))
                        pred = mobilenet.predict(img_tensor)[0][0]
                        if pred > 0.5:
                            st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">✅ ΚΑΤΑΣΤΑΣΗ: ΦΥΣΙΟΛΟΓΙΚΟ (GOOD)</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div style="background-color: green; color: white; padding: 15px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;">❌ ΚΑΤΑΣΤΑΣΗ: ΕΛΑΤΤΩΜΑΤΙΚΟ (ANOMALY)</div>', unsafe_allow_html=True)
                        st.metric("Confidence", f"{(pred * 100 if pred > 0.5 else (1 - pred) * 100):.2f}%")

except Exception as e:
    handle_external_error(e)