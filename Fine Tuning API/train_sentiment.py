# Movie Review Sentiment Classifier using DistilBERT
# pip install transformers datasets evaluate accelerate torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import numpy as np

import torch

# 1. Load dataset
dataset = load_dataset("stanfordnlp/imdb")

# Shuffle and take smaller subset
train_dataset = dataset["train"].shuffle(seed=42)
test_dataset = dataset["test"].shuffle(seed=42)

small_train = train_dataset.select(range(1000))
small_test = test_dataset.select(range(200))

# 2. Load tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize text
def preprocess(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_train = small_train.map(preprocess, batched=True)
tokenized_test = small_test.map(preprocess, batched=True)

# Prepare labels
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_test = tokenized_test.rename_column("label", "labels")

# Remove raw text
tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_test = tokenized_test.remove_columns(["text"])

# Convert to torch format
tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

# 3. Load DistilBERT model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# 4. Metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"]
    }

# 5. Training setup
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)

import datasets.config

# Disable torchvision integration in datasets
datasets.config.TORCHVISION_AVAILABLE = False

# 6. Train model
trainer.train()

# Evaluate model
results = trainer.evaluate()
print("\nEvaluation Results:")
print(results)

# 7. Save model
model.save_pretrained("./movie_sentiment_model")
tokenizer.save_pretrained("./movie_sentiment_model")
print("\nModel saved!")

# 8. Reload model
tokenizer = AutoTokenizer.from_pretrained("./movie_sentiment_model")
model = AutoModelForSequenceClassification.from_pretrained("./movie_sentiment_model")
print("\nModel loaded!")

# 9. Prediction loop
label_map = {
    0: "Negative",
    1: "Positive"
}

while True:
    text = input("\nEnter review (type 'quit' to stop): ")
    if text.lower() == "quit":
        break
        
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    probs = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0][prediction].item() * 100
    
    print(f"Prediction: {label_map[prediction]} ({confidence:.2f}%)")