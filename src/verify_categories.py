import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/l3cube_indicnews")

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

mapping = {
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

for language, splits in files.items():

    print("\n" + "=" * 55)
    print(language.upper())
    print("=" * 55)

    for split, filename in splits.items():

        df = pd.read_csv(DATA_DIR / filename)

        df.columns = df.columns.str.strip().str.lower()

        df = df.dropna(subset=["text", "labels"]).copy()

        df["labels"] = df["labels"].str.strip().str.lower()

        df["category"] = df["labels"].map(mapping[language])

        print(f"\n{split.upper()}")

        print("Original records:", len(df))

        print("Selected records:", df["category"].notna().sum())

        print("\nCategory distribution:")

        print(
            df["category"]
            .value_counts()
            .reindex(["Sports", "Technology", "Entertainment"], fill_value=0)
        )

        print("\nUnselected records:", df["category"].isna().sum())

print("\nCATEGORY MAPPING VERIFICATION COMPLETE!")
