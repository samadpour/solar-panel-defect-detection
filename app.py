import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Perovskite Defect Inspection",
    page_icon="🔬",
    layout="centered"
)

IMG_SIZE = 224

@st.cache_resource
def load_model_and_threshold():
    # Reconstruct MobileNetV2 architecture
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None
    )
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.models.Model(base.input, out)

    # Load trained model weights
    weights_path = "best_model_weights.h5"
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        st.error(f"Weights file ('{weights_path}') not found!")

    # Load decision threshold
    threshold_path = "threshold.txt"
    threshold = 0.5
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            threshold = float(f.read().strip())

    return model, threshold

# Load model and threshold
model, threshold = load_model_and_threshold()

# Header Section
st.title("🔬 Automated Defect Inspection System")
st.write(
    "Upload a characterization image to evaluate defect probability "
    "and classify device health status."
)

st.markdown("---")

# File Upload Section
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    # Display Input Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Characterization Image", use_container_width=True)

    # Image Preprocessing
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model Inference
    with st.spinner("Analyzing image features..."):
        prob = float(model.predict(img_array, verbose=0)[0][0])

    is_defect = prob > threshold

    st.markdown("---")
    st.subheader("Evaluation Results")

    # Classification Display
    if is_defect:
        st.error("⚠️ **Status: Defect Detected**")
        st.caption("The image exhibits structural/surface abnormalities exceeding the quality threshold.")
    else:
        st.success("✅ **Status: Healthy (No Defect)**")
        st.caption("The image passes quality checks with no significant defects detected.")

    # Metric Columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Defect Probability", value=f"{prob:.2%}")
    with col2:
        st.metric(label="Decision Threshold", value=f"{threshold:.2%}")

    # Visual Gauge Bar
    st.write("**Probability Score Indicator:**")
    st.progress(prob)
