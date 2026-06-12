from fastapi import FastAPI
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch
import logging
import time


# Logging


logging.basicConfig(
    level=logging.INFO
)


# FastAPI App


app = FastAPI(
    title="Transformer Spam Detection API"
)


# Load Model + Tokenizer


MODEL_PATH = "model/transformer"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

logging.info(
    "Transformer model loaded!"
)


# Health Check


@app.get("/")
def home():

    return {
        "message":
        "Transformer Spam API Running"
    }

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# Prediction Route


@app.post("/predict")
def predict(data: dict):

    start_time = time.time()

    try:

        
        # Validation
        

        if "text" not in data:

            return {
                "error":
                "Missing 'text' field"
            }

        text = data["text"]

        if not isinstance(
            text,
            str
        ):

            return {
                "error":
                "Text must be string"
            }

        if not text.strip():

            return {
                "error":
                "Input text cannot be empty"
            }

        
        # Logging
        

        logging.info(
            f"Request: {text}"
        )

        
        # Tokenization
        

        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        
        # Prediction
        

        with torch.no_grad():

            outputs = model(
                **inputs
            )

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = torch.max(
            probabilities
        ).item()

        label = (
            "spam"
            if prediction == 1
            else "not_spam"
        )

        latency = (
            time.time()
            - start_time
        )

        logging.info(
            f"Prediction: {label}"
        )

        logging.info(
            f"Latency: {latency:.4f}"
        )

        return {
            "prediction": label,
            "confidence":
            round(confidence, 4),
            "latency_seconds":
            round(latency, 4)
        }

    except Exception as e:

        logging.error(str(e))

        return {
            "error": str(e)
        }