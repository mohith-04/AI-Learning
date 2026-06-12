from fastapi import FastAPI
import joblib
import logging
import time


# Logging Configuration


logging.basicConfig(level=logging.INFO)


# Create FastAPI App


app = FastAPI(
    title="Spam Detection API"
)


# Load Saved Model


model = joblib.load(
    "model/baseline/spam_model.pkl"
)

vectorizer = joblib.load(
    "model/baseline/vectorizer.pkl"
)

logging.info("Model loaded successfully!")


# Health Check Route


@app.get("/")
def home():

    return {
        "message": "Spam Detection API Running"
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

        if not isinstance(text, str):
            return {
                "error":
                "Text must be string"
            }

        if not text.strip():
            return {
                "error":
                "Input text cannot be empty"
            }

        
        # Logging Request
        

        logging.info(
            f"Request received: {text}"
        )

        
        # Vectorize Input
        

        text_vector = vectorizer.transform(
            [text]
        )

        
        # Prediction
        

        prediction = model.predict(
            text_vector
        )[0]

        probabilities = model.predict_proba(
            text_vector
        )[0]

        confidence = max(
            probabilities
        )

        label = (
            "spam"
            if prediction == 1
            else "not_spam"
        )

        
        # Latency
        

        latency = (
            time.time()
            - start_time
        )

        logging.info(
            f"Prediction: {label}"
        )

        logging.info(
            f"Latency: {latency:.4f} sec"
        )

        
        # Response
        

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

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": True
    }