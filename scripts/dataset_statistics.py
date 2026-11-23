
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "raw"
    / "20_newsgroups"
    / "20_newsgroups.csv"
)

REPORT_DIR = BASE_DIR / "datasets" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_PATH)

# Calculate document statistics
df["character_count"] = df["text"].str.len()

df["word_count"] = df["text"].str.split().str.len()

print("TOTAL DOCUMENTS:", len(df))

print("\nDOCUMENT LENGTH STATISTICS")

print(df["word_count"].describe())

print("\nCATEGORY DISTRIBUTION")

category_counts = df["category"].value_counts()

print(category_counts)

# Save statistics
statistics = df["word_count"].describe()

statistics.to_csv(
    REPORT_DIR / "document_statistics.csv"
)

category_counts.to_csv(
    REPORT_DIR / "category_distribution.csv"
)

# Category distribution chart
plt.figure(figsize=(14, 7))

category_counts.plot(kind="bar")

plt.title("20 Newsgroups - Category Distribution")

plt.xlabel("Category")

plt.ylabel("Number of Documents")

plt.xticks(rotation=75)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "category_distribution.png",
    dpi=300
)

plt.show()

# Document length distribution
plt.figure(figsize=(10, 6))

plt.hist(
    df["word_count"],
    bins=50
)

plt.title("Document Word Count Distribution")

plt.xlabel("Number of Words")

plt.ylabel("Number of Documents")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "document_length_distribution.png",
    dpi=300
)

plt.show()

print("\nReports generated successfully.")
