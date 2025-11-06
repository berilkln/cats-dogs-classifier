import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("models/cats_dogs_classification_model.keras")

def predict(img):
    img = img.resize((224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]

    if 0.1 <= pred <= 0.55:
        return "❌ Not a cat or dog"
    return "🐶 Dog" if pred > 0.5 else "🐱 Cat"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🐾 Cats vs Dogs Classifier",
    description="Upload an image to classify whether it's a cat 🐱 or a dog 🐶"
)

demo.launch()


