import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np


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


def predict_issue(text):
    disease = nlp_model.predict([text])[0]
    row = nlp_df[nlp_df['disease'] == disease].iloc[0]
    return {
        'Disease': row['disease'],
        'Cause': row.get('cause', 'Not available'),
        'Symptoms': row.get('symptoms', 'Not available'),
        'Prevention': row.get('prevention', 'Not available')
    }


# ------------------- Streamlit Layout -------------------
st.set_page_config(page_title="Plant Disease Detection & Chatbot", layout="wide")
st.title("🌿 Plant Disease Detection & Crop Chatbot")

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
# ------------------- Tab 2: NLP Chatbot Interface -------------------
with tab2:
    st.subheader("🌱 Crop Chatbot")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_input = st.text_input("Enter your crop issue:")

    if st.button("Send") and user_input.strip() != "":
        # Get prediction
        result = predict_issue(user_input)

        # Format bot response
        bot_response = f"**Disease:** {result['Disease']}\n- **Cause:** {result['Cause']}\n- **Symptoms:** {result['Symptoms']}\n- **Prevention:** {result['Prevention']}"

        # Save messages to chat history
        st.session_state.chat_history.append({"user": user_input, "bot": bot_response})

    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown("---")
