

# Spam Detection API (Baseline + Transformer)

A production-style spam detection API built using both **Traditional Machine Learning** and **Transformer-based NLP** approaches.

This project compares:

* **TF-IDF + Logistic Regression** (Baseline)
* **DistilBERT Transformer Model**

Both models are deployed using **FastAPI** with structured JSON outputs, confidence scores, validation, and API testing.

---

## Project Overview

This project detects whether a message is:

* **Spam**
* **Not Spam**

The goal was to compare a lightweight traditional ML pipeline with a transformer-based NLP model and understand the tradeoffs between:

* Accuracy
* Latency
* Cost
* Inference Speed
* Complexity

---

## Features

### Baseline API

* TF-IDF Vectorization
* Logistic Regression Classifier
* Fast inference
* Low latency
* Lightweight deployment

### Transformer API

* DistilBERT Fine-Tuned Model
* Better contextual understanding
* Confidence scoring
* Structured outputs
* Production-style validation

### API Features

* FastAPI backend
* JSON input/output
* Confidence score
* Error handling
* Input validation
* Logging
* Swagger API testing

---

## Project Structure

```text
spam-detection-project/
│── data/
│   └── spam.csv
│
│── model/
│   ├── baseline/
│   └── transformer/
│
│── baseline/
│   ├── train.py
│   └── main.py
│
│── transformer/
│   ├── train.py
│   └── main.py
│
│── requirements.txt
│── README.md
```

---

## Tech Stack

* Python
* FastAPI
* Scikit-Learn
* TF-IDF
* Logistic Regression
* Transformers
* DistilBERT
* PyTorch
* Postman
* Swagger UI

---

## API Endpoint

### POST `/predict`

### Request Example

```json
{
  "text": "Congratulations! You won ₹50,000!"
}
```

### Response Example

```json
{
  "prediction": "spam",
  "confidence": 0.9821,
  "latency_seconds": 0.0412
}
```

---

## Run the Project

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd spam-detection-project
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate virtual environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Baseline API

```bash
uvicorn baseline.main:app --reload
```

### 5. Run Transformer API

```bash
uvicorn transformer.main:app --reload --port 8001
```

---

## API Testing

Swagger UI:

Baseline API:

```text
http://127.0.0.1:8000/docs
```

Transformer API:

```text
http://127.0.0.1:8001/docs
```

---

## Key Learning Outcomes

* NLP text classification
* TF-IDF vectorization
* Logistic Regression baseline
* Transformer fine-tuning
* FastAPI model serving
* Structured outputs
* API testing
* Validation and error handling
* Production AI mindset

---

## Future Improvements

* Docker deployment
* Cloud deployment
* Authentication
* Database logging
* Model monitoring
* Better evaluation dashboard

```
```
