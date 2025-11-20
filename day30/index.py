import streamlit as st
import requests

# Backend URL
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Nepali Text Classifier", layout="centered")

st.title("🇳🇵 Nepali Text Classifier")
st.write("Enter a Nepali sentence and get the model's prediction.")

text = st.text_area("Enter Nepali sentence:", height=150)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter a sentence.")
    else:
        try:
            # Send request to backend
            response = requests.post(API_URL, json={"text": text})

            if response.status_code == 200:
                data = response.json()
                st.success("Prediction received!")
                st.write("### Model Prediction:")
                st.write(f"**Label_id:** {data['label_id']}")
                st.write(f"**Label:** {data['label']}")
                st.write(f"**Confidence:** {data['confidence']:.4f}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
