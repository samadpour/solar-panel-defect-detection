import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# =========================================================
# CONFIGURATION
# =========================================================

IMG_SIZE = 224
MODEL_PATH = "best_model.keras"
THRESHOLD_PATH = "threshold.txt"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Solar Panel Defect Detection",
    page_icon="☀️",
    layout="centered",
)


# =========================================================
# LOAD MODEL ARCHITECTURE AND WEIGHTS
# =========================================================


@st.cache_resource
def load_model():
    # 1. Instantiate MobileNetV2 backbone without top classifier
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None,
    )

    # 2. Reconstruct the custom classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    # 3. Load weights directly (bypasses broken config/JSON deserialization)
    model.load_weights(MODEL_PATH)
    return model


# =========================================================
# LOAD THRESHOLD
# =========================================================


@st.cache_resource
def load_threshold():
    with open(THRESHOLD_PATH, "r") as f:
        return float(f.read().strip())


# =========================================================
# INITIALIZE
# =========================================================

try:
    model = load_model()
    threshold = load_threshold()

except Exception as e:
    st.error("The AI model could not be loaded.")
    st.exception(e)
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title("☀️ Solar Panel Defect Detection")

st.markdown(
    """
    Upload an image of a solar panel and the deep-learning
    model will classify it as **Reference / No Defect**
    or **Defect Detected**.
    """
)


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("Model information"):

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Architecture:** MobileNetV2")
        st.write(f"**Input size:** {IMG_SIZE} × {IMG_SIZE}")

    with col2:
        st.write("**Task:** Binary classification")
        st.write(f"**Decision threshold:** {threshold:.4f}")

    st.caption("The model was trained using transfer learning and fine-tuning.")


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a solar panel image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG.",
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # LOAD IMAGE
        # -------------------------------------------------

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image, caption="Uploaded image", use_container_width=True
        )

        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------

        img = image.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.asarray(img, dtype=np.float32)

        # Preprocessing scaling [0, 1]
        img_array /= 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        with st.spinner("Analyzing image..."):

            raw_pred = model.predict(img_array, verbose=0)
            probability = float(raw_pred[0][0])

        # -------------------------------------------------
        # CLASSIFICATION
        # -------------------------------------------------

        if probability >= threshold:

            prediction = "DEFECT DETECTED"
            confidence = probability
            is_defect = True

        else:

            prediction = "NO DEFECT DETECTED"
            confidence = 1.0 - probability
            is_defect = False

        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader("Prediction")

        if is_defect:

            st.error(f"⚠️ {prediction}")

        else:

            st.success(f"✓ {prediction}")

        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        st.metric("Prediction confidence", f"{confidence:.1%}")

        # -------------------------------------------------
        # DEFECT PROBABILITY
        # -------------------------------------------------

        st.write(f"**Probability of defect:** {probability:.3f}")

        st.progress(
            probability, text=f"Defect probability: {probability:.1%}"
        )

        # -------------------------------------------------
        # THRESHOLD
        # -------------------------------------------------

        st.caption(f"Decision threshold: {threshold:.3f}")

        if probability >= threshold:

            st.info(
                "The predicted probability is above the deployment threshold."
            )

        else:

            st.info(
                "The predicted probability is below the deployment threshold."
            )

    except Exception as e:

        st.error("An error occurred while processing the image.")

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption("AI-assisted solar panel defect detection using MobileNetV2.")

st.caption("This tool is intended for research and demonstration purposes.")
