
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Initialize NLP resources
stop_words = set(stopwords.words("english"))

custom_stopwords = {
    "subject", "organization", "lines", "article", "writes",
    "university", "edu", "com", "nntp", "posting", "host",
    "thanks", "thank", "know", "think", "like", "just",
    "does", "did", "going", "said", "say", "get", "got",
    "one", "also", "use", "used", "using"
}

stop_words.update(custom_stopwords)

lemmatizer = WordNetLemmatizer()

HEADER_PATTERN = re.compile(
    r"^(from|subject|organization|lines|"
    r"nntp-posting-host|reply-to|distribution|"
    r"keywords|summary|expires|followup-to|"
    r"message-id|references|sender|"
    r"mime-version|content-type):",
    re.IGNORECASE
)


def remove_email_headers(text):
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")

    parts = re.split(r"\n\s*\n", text, maxsplit=1)

    if len(parts) == 2:
        possible_header = parts[0]

        header_lines = possible_header.split("\n")

        matching_headers = sum(
            bool(HEADER_PATTERN.match(line.strip()))
            for line in header_lines
        )

        if matching_headers >= 2:
            return parts[1]

    return text


def remove_quoted_lines(text):
    cleaned_lines = []

    for line in text.split("\n"):
        if re.match(r"^\s*>", line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_text(text):
    """
    Cleaning for BERTopic.
    Stopwords are retained.
    """

    text = remove_email_headers(text)
    text = remove_quoted_lines(text)

    text = text.lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
        " ",
        text
    )

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"[^a-z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text):
    """
    Stopword removal and lemmatization for LDA and NMF.
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


def preprocess_document(text):
    cleaned = clean_text(text)
    processed = preprocess_text(cleaned)

    return {
        "cleaned_text": cleaned,
        "processed_text": processed
    }
