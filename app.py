import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# تنظیمات صفحه Streamlit
st.set_page_config(
    page_title="سیستم تشخیص عیب (Defect Detection)",
    page_icon="🔍",
    layout="centered"
)

IMG_SIZE = 224

@st.cache_resource
def load_model_and_threshold():
    # بازسازی دقیق معماری مدل از کد آموزش
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

    # بارگذاری وزن‌های بهترین مدل
    weights_path = "best_model_weights.h5"
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        st.error(f"فایل وزن‌ها ({weights_path}) یافت نشد!")

    # بارگذاری آستانه بهینه (Threshold)
    threshold_path = "threshold.txt"
    threshold = 0.5
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            threshold = float(f.read().strip())

    return model, threshold

# بارگذاری مدل و آستانه
model, threshold = load_model_and_threshold()

# رابط کاربری
st.title("🔍 سیستم هوشمند تشخیص عیب")
st.write("تصویر مورد نظر را آپلود کنید تا وضعیت آن (**Reference** یا **Defect**) مشخص شود.")

uploaded_file = st.file_uploader(
    "انتخاب تصویر...", 
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    # نمایش تصویر آپلود شده
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="تصویر ورودی", use_container_width=True)

    # پیش‌پردازش تصویر دقیقاً مطابق کد آموزش
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # اجرای پیش‌بینی
    with st.spinner("در حال تحلیل تصویر..."):
        prob = float(model.predict(img_array, verbose=0)[0][0])

    is_defect = prob > threshold
    pred_label = "Defect" if is_defect else "Reference"

    st.markdown("---")
    st.subheader("نتیجه ارزیابی:")

    if is_defect:
        st.error(f"❌ **نتیجه:** {pred_label}")
    else:
        st.success(f"✅ **نتیجه:** {pred_label}")

    st.write(f"**احتمال معیوب بودن (Defect Probability):** `{prob:.4f}`")
    st.write(f"**آستانه تصمیم‌گیری (Threshold):** `{threshold:.4f}`")
    st.progress(prob)
