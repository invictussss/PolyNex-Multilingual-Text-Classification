import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

train = pd.read_csv(DATA_DIR / "train_clean.csv")
test = pd.read_csv(DATA_DIR / "test_clean.csv")
valid = pd.read_csv(DATA_DIR / "valid_clean.csv")


def get_keys(df):
    return set(zip(df["language"], df["text"]))


train_keys = get_keys(train)
test_keys = get_keys(test)
valid_keys = get_keys(valid)

print("=" * 50)
print("DATA LEAKAGE CHECK")
print("=" * 50)

print("Train-Test overlap:", len(train_keys & test_keys))
print("Train-Validation overlap:", len(train_keys & valid_keys))
print("Test-Validation overlap:", len(test_keys & valid_keys))

if not (train_keys & test_keys or
        train_keys & valid_keys or
        test_keys & valid_keys):
    print("\nSUCCESS: No data leakage detected!")
else:
    print("\nWARNING: Duplicate headlines still exist!")

print("\nDATA LEAKAGE CHECK COMPLETED!")
