import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
valid = pd.read_csv(DATA_DIR / "valid.csv")


def get_keys(df):
    return set(zip(df["language"], df["text"]))


train_keys = get_keys(train)

# Remove training duplicates from validation
valid_clean = valid[
    ~valid.apply(
        lambda row: (row["language"], row["text"]) in train_keys,
        axis=1
    )
].copy()

valid_keys = get_keys(valid_clean)

# Remove training and validation duplicates from test
test_clean = test[
    ~test.apply(
        lambda row: (row["language"], row["text"]) in train_keys
        or (row["language"], row["text"]) in valid_keys,
        axis=1
    )
].copy()

# Save cleaned datasets
train.to_csv(DATA_DIR / "train_clean.csv", index=False, encoding="utf-8")
valid_clean.to_csv(DATA_DIR / "valid_clean.csv", index=False, encoding="utf-8")
test_clean.to_csv(DATA_DIR / "test_clean.csv", index=False, encoding="utf-8")

print("=" * 50)
print("DATA LEAKAGE REMOVAL COMPLETED")
print("=" * 50)

print("\nOriginal records:")
print("Train:", len(train))
print("Validation:", len(valid))
print("Test:", len(test))

print("\nClean records:")
print("Train:", len(train))
print("Validation:", len(valid_clean))
print("Test:", len(test_clean))
