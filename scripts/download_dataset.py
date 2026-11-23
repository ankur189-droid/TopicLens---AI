
from sklearn.datasets import fetch_20newsgroups
from pathlib import Path
import pandas as pd
import json

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "datasets" / "raw"
DATA_DIR = RAW_DIR / "20_newsgroups"
METADATA_DIR = RAW_DIR / "metadata"

DATA_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("TOPICLENS - DATASET DOWNLOAD")
print("=" * 60)

# Download complete dataset
dataset = fetch_20newsgroups(
    subset="all",
    remove=(),
    shuffle=True,
    random_state=42
)

# Convert documents into DataFrame
df = pd.DataFrame({
    "document_id": range(1, len(dataset.data) + 1),
    "text": dataset.data,
    "category_id": dataset.target,
    "category": [
        dataset.target_names[i]
        for i in dataset.target
    ]
})

# Save original documents
output_file = DATA_DIR / "20_newsgroups.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)

# Save category metadata
metadata = {
    "dataset_name": "20 Newsgroups",
    "total_documents": len(df),
    "total_categories": len(dataset.target_names),
    "categories": dataset.target_names,
    "random_state": 42
}

metadata_file = METADATA_DIR / "dataset_metadata.json"

with open(metadata_file, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\nDOWNLOAD COMPLETED")

print("Total documents:", len(df))
print("Total categories:", len(dataset.target_names))

print("\nDataset location:")
print(output_file)

print("\nMetadata location:")
print(metadata_file)

print("\nCategory distribution:")
print(df["category"].value_counts())

print("=" * 60)
