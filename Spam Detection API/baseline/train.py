import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# Load Dataset


print("Loading dataset...")

df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'text']

# Convert labels
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("Dataset loaded successfully!")
print(f"Total samples: {len(df)}")



X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42
)

print("\nTrain-test split completed.")


# TF-IDF Vectorization


print("\nVectorizing text using TF-IDF...")

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print("Text vectorization complete.")


# Train Logistic Regression


print("\nTraining model...")

model = LogisticRegression()

model.fit(
    X_train_tfidf,
    y_train
)

print("Model training complete!")


# Predictions


predictions = model.predict(
    X_test_tfidf
)


# Evaluation Metrics


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

print("\n========== MODEL METRICS ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)


# Save Model


joblib.dump(
    model,
    "model/baseline/spam_model.pkl"
)

joblib.dump(
    vectorizer,
    "model/baseline/vectorizer.pkl"
)

print("\nModel saved successfully!")

print(
    "\nSaved files:"
)

print(
    "model/baseline/spam_model.pkl"
)

print(
    "model/baseline/vectorizer.pkl"
)