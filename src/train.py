import pandas as pd
import joblib

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# Project paths
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load cleaned datasets
print("=" * 60)
print("MULTILINGUAL TEXT CLASSIFICATION - MODEL TRAINING")
print("=" * 60)

train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
valid_df = pd.read_csv(DATA_DIR / "valid_clean.csv")

print("\nDataset loaded successfully!")

print("Training records:", len(train_df))
print("Validation records:", len(valid_df))


# Combine language and text
X_train = train_df["language"] + " " + train_df["text"]
y_train = train_df["category"]

X_valid = valid_df["language"] + " " + valid_df["text"]
y_valid = valid_df["category"]


# Create machine learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=300000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42
        )
    )
])


# Train model
print("\nTraining model... Please wait.")

model.fit(X_train, y_train)

print("Model training completed!")


# Validation
print("\nEvaluating model on validation dataset...")

predictions = model.predict(X_valid)

accuracy = accuracy_score(y_valid, predictions)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_valid,
        predictions,
        zero_division=0
    )
)


# Save trained model
model_path = MODEL_DIR / "multilingual_text_classifier.joblib"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Location:", model_path)

print("\nMODEL TRAINING COMPLETED!")
