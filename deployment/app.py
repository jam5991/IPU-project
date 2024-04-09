from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    probabilities: dict

app = FastAPI()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForSequenceClassification.from_pretrained("./model")

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    inputs = tokenizer.encode_plus(request.text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment = "positive" if logits[0, 1] > logits[0, 0] else "negative"
    return {
        "sentiment": sentiment,
        "probabilities": {"negative": probabilities[0], "positive": probabilities[1]}
    }
