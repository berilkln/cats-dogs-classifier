import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("🐾 Cats vs Dogs Image Classifier")

@st.cache_resource
def load_cnn_model():
    model = load_model("models/cats_dogs_classification_model.keras")
    return model

model = load_cnn_model()

st.markdown("## Upload an image of a cat or a dog")
uploaded_file = st.file_uploader(" ", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image", use_container_width=True)
    st.write("")


    img = img.resize((224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if 0.1 <= pred <= 0.55:
        label = "Not a cat or dog (model uncertain)"
    elif pred > 0.5:
        label = "🐶 Dog"
    else:
        label = "🐱 Cat"

    # 🧾 Display result
    st.markdown(f"### **Prediction:** {label}")
    st.caption(f"Model confidence: {pred:.2f}")

