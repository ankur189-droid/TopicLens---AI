
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "raw"
    / "20_newsgroups"
    / "20_newsgroups.csv"
)

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("DATASET INSPECTION")
print("=" * 60)

# Basic information
print("\n1. Dataset shape:")
print(df.shape)

print("\n2. Column names:")
print(df.columns.tolist())

print("\n3. Dataset information:")
print(df.info())

print("\n4. First five records:")
print(df.head())

print("\n5. Category names:")
print(df["category"].unique())

print("\n6. Number of documents per category:")
print(df["category"].value_counts())

print("\n7. Missing values:")
print(df.isnull().sum())

print("\n8. Duplicate documents:")
print(df["text"].duplicated().sum())

print("\n9. Empty documents:")
print((df["text"].fillna("").str.strip() == "").sum())

print("\n10. Sample document:")
print(df.iloc[0]["text"])

print("=" * 60)
