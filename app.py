# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from gradcam import get_gradcam_heatmap, overlay_gradcam

# Load  model
model = tf.keras.models.load_model("model/brain_tumor_model.h5")

# Defining  classes 
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# UI
st.title("🧠 Brain Tumor Detection (MRI)")
st.markdown("Upload an MRI image and let the model detect the type of brain tumor.")

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Preprocessing
    image_np = np.array(image.resize((150, 150))) / 255.0

    if st.button("Predict"):
        # Prediction
        preds = model.predict(np.expand_dims(image_np, axis=0))[0]
        class_idx = np.argmax(preds)
        predicted_class = class_names[class_idx]

        st.success(f"🔍 Prediction: **{predicted_class}**")
        st.bar_chart(preds)

        # GradCAM
        heatmap = get_gradcam_heatmap(model, image_np, last_conv_layer_name='conv5_block3_out')
        gradcam_img = overlay_gradcam(heatmap, (image_np * 255).astype(np.uint8))

        st.image(gradcam_img, caption="Grad-CAM", use_column_width=True)
