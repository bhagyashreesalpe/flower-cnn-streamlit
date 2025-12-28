import os 
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np


model = load_model('Flower_Recog_model.h5')



st.header('Flower Classification CNN model')

flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])

    outcome = (
        'The image belongs to '
        + flower_names[np.argmax(result)]
        + ' with a score of '
        + str(float(np.max(result) * 100))
    )
    return outcome

uploaded_file = st.file_uploader('Upload an image', type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    os.makedirs("upload", exist_ok=True)

    image_path = os.path.join("upload", uploaded_file.name)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, width=200)

    # ✅ ONLY THIS LINE IS IMPORTANT
    st.markdown(classify_images(image_path))
