import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model('mask_detector_model.h5')

st.set_page_config(page_title='Face Mask Detection',layout='centered')
st.title('Face Mask Detection')
st.write('Upload an image and the model will predict whether the person is wearing a mask or not.')

uploaded_file = st.file_uploader(
    'Upload an image', type=['jpg','jpeg','png']
)

def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image,(128,128))
    image = image/255
    image = np.reshape(image, (1,128,128,3))
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded image', use_column_width=400)

    preprocessed_image = preprocess_image(image)

    prediction = model.predict(preprocessed_image)
    prediction_label = np.argmax(prediction)

    st.subheader('Prediction Result')

    if prediction_label == 1:
        st.success('The person is Wearing a mask')
    else:
        st.error('The person is not Wearing a mask')