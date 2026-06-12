import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

import evaluate


# Load Dataset


print("Loading dataset...")

df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)

df = df[['v1', 'v2']]
df.columns = ['label', 'text']

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("Dataset loaded!")


# Convert to HuggingFace Dataset


dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

train_dataset = dataset["train"]
test_dataset = dataset["test"]


# Load Tokenizer


model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


# Tokenization Function


def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# Load Model


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# Metrics


accuracy_metric = evaluate.load(
    "accuracy"
)

f1_metric = evaluate.load(
    "f1"
)

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )

    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels
    )

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }


# Training Arguments


training_args = TrainingArguments(
    output_dir="transformer/results",

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_dir="transformer/logs"
)


# Trainer


trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics
)


# Train


print("Training started...")

trainer.train()


# Evaluate


results = trainer.evaluate()

print("\nEvaluation Results:")
print(results)


# Save Model


model.save_pretrained(
    "model/transformer"
)

tokenizer.save_pretrained(
    "model/transformer"
)

print("\nTransformer model saved!")