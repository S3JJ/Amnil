## Nepali Sentiment Analysis using BERT
---
This folder contains the notebook used to finetune NepaliBERT, FastAPI backend and a simple Streamlit frontend.
It also contains the trained model in ONNX format.
---
### 📖 How to run the model?

1. Install the required packages
   
    `pip install -r requirements.txt`

2. Create a folder named `final_model_onnx` inside `models` folder and paste the model files inside.

3. Run the FastAPI backend using uvicorn
   
    `uvicorn app:app --reload`

4. Run the Streamlit frontend
   
    `streamlit run index.py`

