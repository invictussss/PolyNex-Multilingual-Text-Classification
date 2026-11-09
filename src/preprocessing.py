import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path("data/raw/l3cube_indicnews")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "English": {
        "train": "English/SHC/English_train_shc.csv",
        "test": "English/SHC/English_test_shc.csv",
        "valid": "English/SHC/English_valid_shc.csv"
    },

    "Hindi": {
        "train": "Hindi/SHC/Hindi_SHC_Train.csv",
        "test": "Hindi/SHC/Hindi_SHC_Test.csv",
        "valid": "Hindi/SHC/Hindi_SHC_Valid.csv"
    },

    "Bengali": {
        "train": "Bengali/SHC/Bengal_SHC_Train.csv",
        "test": "Bengali/SHC/Bengal_SHC_Test.csv",
        "valid": "Bengali/SHC/Bengal_SHC_Valid.csv"
    },

    "Tamil": {
        "train": "Tamil/SHC/Tamil_SHC_Train.csv",
        "test": "Tamil/SHC/Tamil_SHC_Test.csv",
        "valid": "Tamil/SHC/Tamil_SHC_Valid.csv"
    }
}

category_mapping = {
    "English": {
        "sports": "Sports",
        "technology": "Technology",
        "entertainment": "Entertainment"
    },

    "Hindi": {
        "khel": "Sports",
        "technology-news": "Technology",
        "entertainment": "Entertainment"
    },

    "Bengali": {
        "sports": "Sports",
        "technology": "Technology",
        "entertainment": "Entertainment"
    },

    "Tamil": {
        "sports": "Sports",
        "technology": "Technology",
        "cinema": "Entertainment"
    }
}


def clean_text(text):

    text = str(text)

    # Normalize whitespace without removing language characters
    text = re.sub(r"\s+", " ", text)

    return text.strip()


all_datasets = {
    "train": [],
    "test": [],
    "valid": []
}


for language, splits in files.items():

    print(f"\nProcessing {language} dataset...")

    for split, filename in splits.items():

        filepath = DATA_DIR / filename

        df = pd.read_csv(filepath)

        df.columns = df.columns.str.strip().str.lower()

        df = df.dropna(subset=["text", "labels"]).copy()

        df["text"] = df["text"].astype(str).apply(clean_text)

        df["labels"] = (
            df["labels"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["category"] = df["labels"].map(
            category_mapping[language]
        )

        df = df.dropna(subset=["category"])

        df["language"] = language

        df = df[["text", "language", "category"]]

        df = df[df["text"].str.len() > 0]

        df = df.drop_duplicates(subset=["text"])

        all_datasets[split].append(df)

        print(
            f"{split.upper()}: "
            f"{len(df)} records processed"
        )


for split, datasets in all_datasets.items():

    final_df = pd.concat(
        datasets,
        ignore_index=True
    )

    output_file = OUTPUT_DIR / f"{split}.csv"

    final_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print(f"\nSaved: {output_file}")
    print("Total records:", len(final_df))

    print("\nLanguage distribution:")
    print(final_df["language"].value_counts())

    print("\nCategory distribution:")
    print(final_df["category"].value_counts())


print("\n===================================")
print("DATASET PREPROCESSING COMPLETED!")
print("===================================")