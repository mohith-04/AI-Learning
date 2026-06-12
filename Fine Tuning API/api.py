from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Initialize FastAPI
app = FastAPI(
    title="DistilBERT Sentiment Analysis API",
    description="A simple API to classify movie review sentiment using a fine-tuned DistilBERT model."
)

# 2. Load the fine-tuned model and tokenizer from your local directory
MODEL_PATH = "./movie_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Set model to evaluation mode for faster/consistent inference
model.eval()

# Label mapping
LABELS = {
    0: "negative",
    1: "positive"
}

# 3. Define the expected request body structure using Pydantic
class ReviewRequest(BaseModel):
    text: str

# 4. Define the API endpoint
@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    # Extract text safely from the request object
    text = request.text

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Perform inference (turn off gradient calculation to save memory)
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert raw logits to probabilities
    probs = torch.softmax(outputs.logits, dim=1)

    # Extract confidence score and prediction ID
    confidence = probs.max().item()
    pred_id = probs.argmax().item()

    # Return structured JSON response
    return {
        "text_analyzed": text,
        "prediction": LABELS[pred_id],
        "confidence": round(confidence, 4)
    }