
import json
import re
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    BASE_DIR
    / "datasets"
    / "raw"
    / "20_newsgroups"
    / "20_newsgroups.csv"
)

OUTPUT_DIR = BASE_DIR / "datasets" / "processed"

OUTPUT_PATH = OUTPUT_DIR / "20_newsgroups_preprocessed.csv"
REPORT_PATH = OUTPUT_DIR / "preprocessing_report.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. NLP INITIALIZATION
# --------------------------------------------------

print("=" * 65)
print("TOPICLENS AI - NLP TEXT PREPROCESSING")
print("=" * 65)

stop_words = set(stopwords.words("english"))

# Add common words and artifacts that are not useful
# for identifying the main topic of a document.
custom_stopwords = {
    "subject",
    "organization",
    "lines",
    "article",
    "writes",
    "university",
    "edu",
    "com",
    "nntp",
    "posting",
    "host",
    "thanks",
    "thank",
    "know",
    "think",
    "like",
    "just",
    "does",
    "did",
    "going",
    "said",
    "say",
    "get",
    "got",
    "one",
    "also",
    "use",
    "used",
    "using",
}

stop_words.update(custom_stopwords)

lemmatizer = WordNetLemmatizer()


# --------------------------------------------------
# 3. REMOVE EMAIL HEADERS
# --------------------------------------------------

HEADER_PATTERN = re.compile(
    r"^(from|subject|organization|lines|"
    r"nntp-posting-host|reply-to|distribution|"
    r"keywords|summary|expires|followup-to|"
    r"message-id|references|sender|"
    r"mime-version|content-type):",
    re.IGNORECASE
)


def remove_email_headers(text):
    """
    Remove the email/news article header section.
    Preserve the body of the document.
    """

    text = str(text).replace("\r\n", "\n").replace("\r", "\n")

    parts = re.split(r"\n\s*\n", text, maxsplit=1)

    if len(parts) == 2:
        possible_header = parts[0]

        header_lines = possible_header.split("\n")

        matching_headers = sum(
            bool(HEADER_PATTERN.match(line.strip()))
            for line in header_lines
        )

        # Remove the first section only if it looks
        # like a genuine email/news header.
        if matching_headers >= 2:
            return parts[1]

    return text


# --------------------------------------------------
# 4. REMOVE QUOTED LINES
# --------------------------------------------------

def remove_quoted_lines(text):
    """
    Remove lines that quote another person's message.
    """

    cleaned_lines = []

    for line in text.split("\n"):
        if re.match(r"^\s*>", line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# --------------------------------------------------
# 5. CLEAN AND NORMALIZE TEXT
# --------------------------------------------------

def clean_text(text):
    """
    Basic text cleaning for BERTopic.
    Stopwords are intentionally retained here.
    """

    text = remove_email_headers(text)
    text = remove_quoted_lines(text)

    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Keep alphabetic words only
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# 6. REMOVE STOPWORDS AND LEMMATIZE
# --------------------------------------------------

def preprocess_text(text):
    """
    Create processed text for LDA and NMF.
    """

    words = re.findall(r"\b[a-z]+\b", text)

    processed_words = []

    for word in words:

        if word in stop_words:
            continue

        if len(word) < 2:
            continue

        lemma = lemmatizer.lemmatize(word)

        if lemma not in stop_words:
            processed_words.append(lemma)

    return " ".join(processed_words)


# --------------------------------------------------
# 7. LOAD DATASET
# --------------------------------------------------

print("\nLoading original dataset...")

df = pd.read_csv(INPUT_PATH)

print("Original document count:", len(df))


# --------------------------------------------------
# 8. APPLY PREPROCESSING
# --------------------------------------------------

print("\nRemoving headers, quotes and unwanted characters...")

tqdm.pandas()

df["cleaned_text"] = df["text"].progress_apply(clean_text)

print("\nRemoving stopwords and applying lemmatization...")

df["processed_text"] = df["cleaned_text"].progress_apply(
    preprocess_text
)


# --------------------------------------------------
# 9. CALCULATE DOCUMENT STATISTICS
# --------------------------------------------------

df["original_word_count"] = df["text"].str.split().str.len()

df["cleaned_word_count"] = (
    df["cleaned_text"].str.split().str.len()
)

df["processed_word_count"] = (
    df["processed_text"].str.split().str.len()
)

empty_cleaned = int(
    (df["cleaned_text"].str.strip() == "").sum()
)

empty_processed = int(
    (df["processed_text"].str.strip() == "").sum()
)


# --------------------------------------------------
# 10. SAVE PROCESSED DATASET
# --------------------------------------------------

print("\nSaving preprocessed dataset...")

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)


# --------------------------------------------------
# 11. GENERATE PREPROCESSING REPORT
# --------------------------------------------------

report = {
    "dataset_name": "20 Newsgroups",
    "total_documents": int(len(df)),
    "total_categories": int(df["category"].nunique()),

    "preprocessing_steps": [
        "Email header removal",
        "Quoted line removal",
        "Lowercase conversion",
        "URL removal",
        "Email address removal",
        "HTML tag removal",
        "Special character removal",
        "Whitespace normalization",
        "Stopword removal",
        "Lemmatization"
    ],

    "empty_cleaned_documents": empty_cleaned,
    "empty_processed_documents": empty_processed,

    "average_original_words": round(
        float(df["original_word_count"].mean()), 2
    ),

    "average_cleaned_words": round(
        float(df["cleaned_word_count"].mean()), 2
    ),

    "average_processed_words": round(
        float(df["processed_word_count"].mean()), 2
    ),

    "original_total_words": int(
        df["original_word_count"].sum()
    ),

    "processed_total_words": int(
        df["processed_word_count"].sum()
    )
}

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)


# --------------------------------------------------
# 12. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("PREPROCESSING COMPLETED")
print("=" * 65)

print("Total documents:", len(df))
print("Total categories:", df["category"].nunique())

print("\nEmpty cleaned documents:", empty_cleaned)
print("Empty processed documents:", empty_processed)

print("\nAverage word count:")
print("Original:", report["average_original_words"])
print("Cleaned:", report["average_cleaned_words"])
print("Processed:", report["average_processed_words"])

print("\nProcessed dataset saved at:")
print(OUTPUT_PATH)

print("\nPreprocessing report saved at:")
print(REPORT_PATH)

print("\n" + "=" * 65)
print("SAMPLE DOCUMENT COMPARISON")
print("=" * 65)

sample = df.iloc[0]

print("\nORIGINAL TEXT:\n")
print(sample["text"][:1000])

print("\nCLEANED TEXT:\n")
print(sample["cleaned_text"][:1000])

print("\nPROCESSED TEXT:\n")
print(sample["processed_text"][:1000])

print("\n" + "=" * 65)
