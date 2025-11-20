from fastapi import FastAPI
from pydantic import BaseModel
from models.model import sentiment_model

app = FastAPI(title="Nepali Sentiment Analysis API")

class InputText(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Nepali Sentiment API is running"}


@app.post("/predict")
def predict_sentiment(data: InputText):
    """
    Takes a Nepali sentence as an input and returns inference result in JSON format containing label_id, 
    label, confidence and logits.
    """
    result = sentiment_model.predict(data.text)
    return result

