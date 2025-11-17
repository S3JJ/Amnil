import streamlit as st
import requests
from PIL import Image
import io

# Backend API URL
API_URL = "http://127.0.0.1:8000/predict"  

st.set_page_config(page_title="Image Classification using EfficientNet-B0", layout="wide")

_,col1,_=st.columns([1,4,1])
with col1:
  st.title("Image Classification using EfficientNet-B0")
  st.text("")
  st.write("Upload an image and get its predicted class.")

  # File uploader for uploading the image
  uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

  if uploaded_file is not None:
      # Displaying the uploaded image
      image = Image.open(uploaded_file)
      with st.container(horizontal_alignment="center", horizontal=True):
        st.image(image, caption="Uploaded Image", use_container_width=False, width=800)

      # Adding a predict button
      if st.button("Predict", type="primary", use_container_width=True):
          # Converting the image to bytes for sending
          img_bytes = io.BytesIO()
          image.save(img_bytes, format="JPEG")
          img_bytes = img_bytes.getvalue()

          # Sending image to FastAPI backend
          with st.spinner("Predicting..."):
              try:
                  response = requests.post(
                      API_URL,
                      files={"file": ("image.jpg", img_bytes, "image/jpeg")}
                  )

                  if response.status_code == 200:
                      result = response.json()
                      st.success("Prediction Successful!")
                      st.write(f"**Class:** {result['class_name']}")
                      st.write(f"**WNID:** {result['wnid']}")
                      st.write(f"**Confidence:** {result['confidence']*100:.2f}%")
                      st.write(f"**Latency:** {result['latency_ms']:.2f}ms")
                      st.write(f"**Throughput:** {result['throughput_rps']:.2f}rps")
                      st.write(f"**CPU Usage:** {result['cpu_usage_percent']:.2f}%")
                      st.write(f"**RAM Usage:** {result['ram_usage_percent']:.2f}%")

                      gpu_mem = result.get("gpu_memory_mb")

                      if gpu_mem is None:
                          st.write("**GPU Memory (MB):** GPU not found")
                      else:
                          st.write(f"**GPU Memory (MB):** {gpu_mem:.2f} MB")
                      st.divider()
                      st.write(f"**Model Parameters:** {result['model_parameters']}")
                      st.write(f"**Model Size:** {result['model_size_mb']} MB")
                      

                  else:
                      st.error(f"Error: {response.status_code} - {response.text}")
              except Exception as e:
                  st.error(f"Failed to connect to API: {e}")

