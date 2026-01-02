import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Flower Classification", layout="centered")

st.title("🌷 Flower Classification CNN Model")
st.markdown("### 👇 Upload an image to get prediction")

@st.cache_resource(show_spinner=False)
def load_cnn_model():
    return tf.keras.models.load_model("model.keras")

with st.spinner("🔄 Loading CNN model, please wait..."):
    model = load_cnn_model()

flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    st.success(
        f"This image is **{flower_names[np.argmax(score)]}** "
        f"with **{100 * np.max(score):.2f}% confidence**"
    )
