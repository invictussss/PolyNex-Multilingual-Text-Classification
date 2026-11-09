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

for language, splits in files.items():

    print("\n" + "=" * 60)
    print(f"LANGUAGE: {language}")
    print("=" * 60)

    for split, filename in splits.items():

        filepath = DATA_DIR / filename

        df = pd.read_csv(filepath)

        print(f"\n{split.upper()} DATASET")
        print("-" * 40)

        print("Rows:", len(df))
        print("Columns:", list(df.columns))

        print("\nFirst 3 records:")
        print(df.head(3).to_string(index=False))

        print("\nMissing values:")
        print(df.isnull().sum())

        print("\nDataset information:")
        print(df.info())

        print("\n" + "-" * 40)
