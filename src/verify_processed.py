import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

for split in ["train", "test", "valid"]:

    print("\n" + "=" * 55)
    print(f"{split.upper()} DATASET VERIFICATION")
    print("=" * 55)

    df = pd.read_csv(DATA_DIR / f"{split}.csv")

    print("\nTotal records:", len(df))

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate texts:", df["text"].duplicated().sum())

    print("\nDuplicate language-text pairs:",
          df.duplicated(subset=["language", "text"]).sum())

    print("\nLanguage distribution:")
    print(df["language"].value_counts())

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    print("\nLanguage-wise category distribution:")
    print(pd.crosstab(df["language"], df["category"]))

    print("\nEmpty texts:", (df["text"].str.strip() == "").sum())

print("\nDATASET QUALITY CHECK COMPLETED!")