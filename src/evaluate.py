import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("MULTILINGUAL TEXT CLASSIFICATION - TEST EVALUATION")
print("=" * 60)

# Load test dataset
test_df = pd.read_csv(DATA_DIR / "test_clean.csv")

# Load trained model
model = joblib.load(
    MODEL_DIR / "multilingual_text_classifier.joblib"
)

print("\nTest records:", len(test_df))

# Prepare test data
X_test = test_df["language"] + " " + test_df["text"]
y_test = test_df["category"]

# Generate predictions
print("\nEvaluating model on unseen test data...")

predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Classification report
report = classification_report(
    y_test,
    predictions,
    zero_division=0
)

print("\nClassification Report:")
print(report)

# Confusion matrix
labels = sorted(y_test.unique())

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print("\nConfusion Matrix:")
print(pd.DataFrame(
    matrix,
    index=labels,
    columns=labels
))

# Language-wise accuracy
test_df["prediction"] = predictions

language_results = test_df.groupby("language").apply(
    lambda group: accuracy_score(
        group["category"],
        group["prediction"]
    ),
    include_groups=False
)

print("\nLanguage-wise Accuracy:")

for language, score in language_results.items():
    print(f"{language}: {score * 100:.2f}%")

# Save evaluation report
with open(REPORT_DIR / "evaluation_report.txt", "w", encoding="utf-8") as file:
    file.write("MULTILINGUAL TEXT CLASSIFICATION - TEST RESULTS\n\n")
    file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    ).to_string())
    file.write("\n\nLanguage-wise Accuracy:\n")
    file.write(language_results.to_string())

print("\nEvaluation report saved in reports/evaluation_report.txt")

print("\nTEST EVALUATION COMPLETED!")
