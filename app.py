import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image
from keras.models import load_model
import platform

# Configuración de estilo (fondo y color de letras)
page_bg = """
<style>
    .stApp {
        background-color: #936639;
        color: #333d29;
    }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Muestra la versión de Python
st.write("Versión de Python:", platform.python_version())

# Carga del modelo
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Imágenes")

# Texto adicional
st.subheader("Imita una de estas imágenes con tus manos")

# Mostrar imágenes de referencia
col1, col2, col3 = st.columns(3)
with col1:
    st.image("Perro.jpg", width=200)
with col2:
    st.image("Pájaro.jpg", width=200)
with col3:
    st.image("Coraazón.jpg", width=200)

# Sidebar
with st.sidebar:
    st.subheader("Usando un modelo entrenado en Teachable Machine puedes usarlo en esta app para identificar")

# Cámara
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    # Inicializar el array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Leer la imagen como PIL
    img = Image.open(img_file_buffer)

    # Redimensionar
    newsize = (224, 224)
    img = img.resize(newsize)

    # Convertir a numpy
    img_array = np.array(img)

    # Normalizar
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1

    # Cargar en el array
    data[0] = normalized_image_array

    # Inferencia
    prediction = model.predict(data)
    print(prediction)

    if prediction[0][0] > 0.5:
        st.header('Corazón, con Probabilidad: ' + str(prediction[0][0]))
    if prediction[0][1] > 0.5:
        st.header('Perro, con Probabilidad: ' + str(prediction[0][1]))
    if prediction[0][2] > 0.5:
        st.header('Pájaro, con Probabilidad: ' + str(prediction[0][2]))
    if prediction[0][3] > 0.5:
        st.header('Nada, con Probabilidad: ' + str(prediction[0][3]))
