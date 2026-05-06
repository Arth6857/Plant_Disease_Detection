import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

working_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    # Load the fully trained model
    return tf.keras.models.load_model(
        f"{working_dir}/model/plant_disease_model.keras"
    )

@st.cache_resource
def load_classes():
    with open(f"{working_dir}/data/class_indices.json") as f:
        return {int(k): v for k, v in json.load(f).items()}

class_indices = load_classes()

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detector")
st.markdown("Upload a leaf image to detect plant disease instantly")
st.divider()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", width=600)

    if st.button("🔍 Detect Disease", use_container_width=True):
        with st.spinner("Loading model..."):
            model = load_model()

        with st.spinner("Analyzing leaf..."):
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype('float32') / 255.

            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions, axis=1)[0]
            confidence = float(np.max(predictions[0])) * 100
            predicted_class = class_indices[predicted_index]

            if "___" in predicted_class:
                plant, condition = predicted_class.split("___")
            else:
                plant = predicted_class
                condition = "Unknown"

            st.divider()
            if "healthy" in predicted_class.lower():
                st.success("✅ Plant is Healthy!")
            else:
                st.error("⚠️ Disease Detected!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🌱 Plant", plant.replace("_", " "))
            with col2:
                st.metric("🔬 Condition", condition.replace("_", " "))

            st.metric("📊 Confidence", f"{confidence:.2f}%")
