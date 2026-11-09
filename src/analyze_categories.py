import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/l3cube_indicnews")

files = {
    "English": "English/SHC/English_train_shc.csv",
    "Hindi": "Hindi/SHC/Hindi_SHC_Train.csv",
    "Bengali": "Bengali/SHC/Bengal_SHC_Train.csv",
    "Tamil": "Tamil/SHC/Tamil_SHC_Train.csv"
}

all_labels = {}

for language, filename in files.items():

    print("\n" + "=" * 60)
    print(f"{language.upper()} CATEGORY ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(DATA_DIR / filename)

    df.columns = df.columns.str.strip().str.lower()

    df = df.dropna(subset=["text", "labels"])

    df["labels"] = df["labels"].str.strip()

    counts = df["labels"].value_counts()

    all_labels[language] = set(counts.index)

    print("\nTotal records:", len(df))

    print("\nAvailable categories:")
    print(counts.to_string())

    print("\nNumber of categories:", len(counts))

print("\n" + "=" * 60)
print("CATEGORY SUMMARY")
print("=" * 60)

for language, labels in all_labels.items():
    print(f"\n{language}: {len(labels)} categories")

    for label in sorted(labels):
        print(" -", label)