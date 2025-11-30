import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from functools import lru_cache

# ------------------- Load CNN Model -------------------
@st.cache_resource
def load_cnn_model():
    return load_model("plant_model.keras")

@st.cache_resource
def load_class_names():
    try:
        with open("class_names.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    except:
        return None

cnn_model = load_cnn_model()
class_names = load_class_names()
if class_names is None:
    num_classes = cnn_model.output_shape[-1]
    class_names = [f"Class_{i}" for i in range(num_classes)]

# ------------------- Load NLP Model & Dataset -------------------
@st.cache_resource
def load_nlp_model():
    return joblib.load("plantvillage_nlp_model.joblib")

@st.cache_data
def load_nlp_dataset():
    return pd.read_csv("plantvillage_nlp_dataset.csv")

nlp_model = load_nlp_model()
nlp_df = load_nlp_dataset()

# ------------------- Translation Cache -------------------
@lru_cache(maxsize=512)
def translate(text, source_lang, target_lang):
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except:
        return text  # fallback if translation fails

# ------------------- Multilingual NLP Prediction -------------------
def predict_issue_multilingual(text):
    input_lang = detect(text)
    # Translate user input to English for model prediction
    translated_text = text if input_lang == 'en' else translate(text, input_lang, 'en')

    # Predict disease
    disease_en = nlp_model.predict([translated_text])[0]

    # Get info from dataset
    row = nlp_df[nlp_df['disease'] == disease_en].iloc[0]
    cause = row.get('cause', 'Not available')
    symptoms = row.get('symptoms', 'Not available')
    prevention = row.get('prevention', 'Not available')

    # Translate response back to user language
    if input_lang != 'en':
        cause = translate(cause, 'en', input_lang)
        symptoms = translate(symptoms, 'en', input_lang)
        prevention = translate(prevention, 'en', input_lang)
        # Optionally, you can also translate disease name
        disease = translate(disease_en, 'en', input_lang)
    else:
        disease = disease_en

    return {
        "Disease": disease,
        "Cause": cause,
        "Symptoms": symptoms,
        "Prevention": prevention
    }

# ------------------- Streamlit Layout -------------------
st.set_page_config(page_title="Plant Disease Detection & Crop Chatbot", layout="wide")
st.title("🌿 Plant Disease Detection & Multilingual Crop Chatbot")

# ------------------- Tabs for CNN and NLP -------------------
tab1, tab2 = st.tabs(["🔍 Image Upload (CNN)", "💬 Text Input (NLP)"])

# ------------------- Tab 1: CNN Image Classifier -------------------
with tab1:
    st.subheader("Upload Leaf Image for Disease Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Disease"):
            img_resized = img.resize((128, 128))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = cnn_model.predict(img_array)
            predicted_idx = int(np.argmax(prediction, axis=1)[0])
            confidence = float(np.max(prediction) * 100)

            if predicted_idx >= len(class_names):
                st.error("Predicted class index exceeds class_names length. Fix class_names.txt!")
            else:
                class_name = class_names[predicted_idx]
                st.success(f"Prediction: {class_name}")
                st.info(f"Confidence: {confidence:.2f}%")

# ------------------- Tab 2: NLP Text Chatbot -------------------
# ------------------- Tab 2: NLP Text Chatbot -------------------
with tab2:
    st.subheader("🌱 Multilingual Crop Chatbot")

    # User input
    user_input = st.text_input("Enter your crop issue in any Indian language:")

    if st.button("Send") and user_input.strip() != "":
        # Get prediction
        result = predict_issue_multilingual(user_input)

        # Format bot response
        bot_response = f"**Disease:** {result['Disease']}\n- **Cause:** {result['Cause']}\n- **Symptoms:** {result['Symptoms']}\n- **Prevention:** {result['Prevention']}"

        # Display only the current response
        st.markdown(bot_response)
