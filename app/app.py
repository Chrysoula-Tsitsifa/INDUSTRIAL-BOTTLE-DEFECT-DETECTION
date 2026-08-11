# ARTIFACT INITIALIZATION
# Persistent models and reference data from application folder.
@st.cache_resource
def load_artifacts():
    pca = joblib.load('app/pca_model.joblib')
    svm = joblib.load('app/svm_model.joblib')
    golden_ref = np.load('app/golden_reference.npy')
    
    # OPTIMIZED AUTOENCODER LOADING
    # Full architecture and weights extraction from file.
    ae_baseline = tf.keras.models.load_model('app/best_model_optimized.h5')

    # MOBILENETV2 ARCHITECTURE CONSTRUCTION
    # Base model instantiation with custom classification layers.
    base_mobilenet = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    mobilenet = Sequential([
        base_mobilenet,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    mobilenet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # PRE-TRAINED WEIGHTS
    # Integration of parameters from the mobilenet artifact.
    mobilenet.load_weights('app/bottle_model_final_tuned.h5')

    # CLASS LABELS EXTRACTION
    # Reference classes from text file.
    with open('app/labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
        
    return pca, svm, ae_baseline, ae_baseline, 0.05, golden_ref, mobilenet, labels