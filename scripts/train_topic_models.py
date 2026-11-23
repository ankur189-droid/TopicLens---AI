
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import (
    LatentDirichletAllocation,
    NMF
)

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "20_newsgroups_preprocessed.csv"
)

LDA_DIR = BASE_DIR / "models" / "lda"
NMF_DIR = BASE_DIR / "models" / "nmf"

EXPERIMENT_DIR = BASE_DIR / "experiments" / "phase3"
REPORT_DIR = BASE_DIR / "reports" / "phase3"

for directory in [LDA_DIR, NMF_DIR, EXPERIMENT_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. MODEL CONFIGURATION
# --------------------------------------------------

N_TOPICS = 20
TOP_WORDS = 15
RANDOM_STATE = 42

MAX_FEATURES = 20000
MIN_DF = 5
MAX_DF = 0.95


# --------------------------------------------------
# 3. HELPER FUNCTION: EXTRACT TOPIC WORDS
# --------------------------------------------------

def extract_topics(model, feature_names, top_n=15):
    """
    Extract the most important words for every topic.
    """

    topics = []

    for topic_id, components in enumerate(model.components_):

        top_indices = components.argsort()[-top_n:][::-1]

        top_words = [
            feature_names[index]
            for index in top_indices
        ]

        topics.append({
            "topic_id": topic_id,
            "top_words": ", ".join(top_words)
        })

    return pd.DataFrame(topics)


# --------------------------------------------------
# 4. HELPER FUNCTION: SAVE DOCUMENT ASSIGNMENTS
# --------------------------------------------------

def save_document_assignments(
    df,
    topic_scores,
    model_name,
    output_path
):
    """
    Assign each document to its highest-scoring topic.
    """

    dominant_topics = np.argmax(topic_scores, axis=1)

    topic_strength = np.max(topic_scores, axis=1)

    assignments = pd.DataFrame({
        "document_id": df["document_id"].values,
        "category": df["category"].values,
        "dominant_topic": dominant_topics,
        "topic_score": topic_strength
    })

    assignments.to_csv(output_path, index=False)

    print(f"{model_name} document assignments saved:")
    print(output_path)

    return assignments


# --------------------------------------------------
# 5. LOAD AND PREPARE DATASET
# --------------------------------------------------

print("=" * 70)
print("TOPICLENS AI - LDA AND NMF TOPIC MODELLING")
print("=" * 70)

print("\nLoading preprocessed dataset...")

df = pd.read_csv(DATASET_PATH)

print("Original records:", len(df))

# Exclude documents with no usable processed text.
df["processed_text"] = df["processed_text"].fillna("").astype(str)

df = df[
    df["processed_text"].str.strip().ne("")
].copy()

df.reset_index(drop=True, inplace=True)

documents = df["processed_text"].tolist()

print("Documents available for modelling:", len(documents))
print("Documents excluded:", 18846 - len(documents))


# --------------------------------------------------
# 6. LDA: COUNT VECTORIZATION
# --------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1: CREATING DOCUMENT-TERM MATRIX FOR LDA")
print("=" * 70)

count_vectorizer = CountVectorizer(
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    max_df=MAX_DF,
    ngram_range=(1, 2)
)

count_matrix = count_vectorizer.fit_transform(documents)

print("Count matrix shape:", count_matrix.shape)
print("Vocabulary size:", len(count_vectorizer.get_feature_names_out()))


# --------------------------------------------------
# 7. TRAIN LDA MODEL
# --------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2: TRAINING LDA MODEL")
print("=" * 70)

lda_model = LatentDirichletAllocation(
    n_components=N_TOPICS,
    max_iter=10,
    learning_method="batch",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    evaluate_every=-1
)

lda_start = time.time()

lda_document_topics = lda_model.fit_transform(count_matrix)

lda_training_time = time.time() - lda_start

lda_perplexity = lda_model.perplexity(count_matrix)

print("\nLDA training completed.")
print("Training time:", round(lda_training_time, 2), "seconds")
print("LDA perplexity:", round(lda_perplexity, 2))


# --------------------------------------------------
# 8. EXTRACT AND SAVE LDA TOPICS
# --------------------------------------------------

lda_feature_names = count_vectorizer.get_feature_names_out()

lda_topics = extract_topics(
    lda_model,
    lda_feature_names,
    TOP_WORDS
)

lda_topics.to_csv(
    LDA_DIR / "lda_topics.csv",
    index=False
)

print("\nLDA TOPICS:")

for _, row in lda_topics.iterrows():
    print(f"\nTopic {row['topic_id']}:")
    print(row["top_words"])


# --------------------------------------------------
# 9. SAVE LDA DOCUMENT ASSIGNMENTS
# --------------------------------------------------

save_document_assignments(
    df,
    lda_document_topics,
    "LDA",
    EXPERIMENT_DIR / "lda_document_topics.csv"
)


# --------------------------------------------------
# 10. SAVE LDA MODEL AND VECTORIZER
# --------------------------------------------------

joblib.dump(
    lda_model,
    LDA_DIR / "lda_model.joblib"
)

joblib.dump(
    count_vectorizer,
    LDA_DIR / "count_vectorizer.joblib"
)

print("\nLDA model and vectorizer saved.")


# --------------------------------------------------
# 11. NMF: TF-IDF VECTORIZATION
# --------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3: CREATING TF-IDF MATRIX FOR NMF")
print("=" * 70)

tfidf_vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    max_df=MAX_DF,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("Vocabulary size:", len(tfidf_vectorizer.get_feature_names_out()))


# --------------------------------------------------
# 12. TRAIN NMF MODEL
# --------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4: TRAINING NMF MODEL")
print("=" * 70)

nmf_model = NMF(
    n_components=N_TOPICS,
    init="nndsvda",
    solver="cd",
    max_iter=400,
    random_state=RANDOM_STATE
)

nmf_start = time.time()

nmf_document_topics = nmf_model.fit_transform(tfidf_matrix)

nmf_training_time = time.time() - nmf_start

nmf_reconstruction_error = nmf_model.reconstruction_err_

print("\nNMF training completed.")
print("Training time:", round(nmf_training_time, 2), "seconds")
print(
    "NMF reconstruction error:",
    round(nmf_reconstruction_error, 4)
)


# --------------------------------------------------
# 13. EXTRACT AND SAVE NMF TOPICS
# --------------------------------------------------

nmf_feature_names = tfidf_vectorizer.get_feature_names_out()

nmf_topics = extract_topics(
    nmf_model,
    nmf_feature_names,
    TOP_WORDS
)

nmf_topics.to_csv(
    NMF_DIR / "nmf_topics.csv",
    index=False
)

print("\nNMF TOPICS:")

for _, row in nmf_topics.iterrows():
    print(f"\nTopic {row['topic_id']}:")
    print(row["top_words"])


# --------------------------------------------------
# 14. SAVE NMF DOCUMENT ASSIGNMENTS
# --------------------------------------------------

save_document_assignments(
    df,
    nmf_document_topics,
    "NMF",
    EXPERIMENT_DIR / "nmf_document_topics.csv"
)


# --------------------------------------------------
# 15. SAVE NMF MODEL AND VECTORIZER
# --------------------------------------------------

joblib.dump(
    nmf_model,
    NMF_DIR / "nmf_model.joblib"
)

joblib.dump(
    tfidf_vectorizer,
    NMF_DIR / "tfidf_vectorizer.joblib"
)

print("\nNMF model and vectorizer saved.")


# --------------------------------------------------
# 16. SAVE MODEL METRICS
# --------------------------------------------------

metrics = {
    "dataset": "20 Newsgroups",
    "total_original_documents": 18846,
    "documents_used": int(len(df)),
    "documents_excluded": int(18846 - len(df)),
    "number_of_topics": N_TOPICS,
    "top_words_per_topic": TOP_WORDS,

    "lda": {
        "training_time_seconds": round(lda_training_time, 2),
        "perplexity": round(float(lda_perplexity), 4)
    },

    "nmf": {
        "training_time_seconds": round(nmf_training_time, 2),
        "reconstruction_error": round(
            float(nmf_reconstruction_error), 4
        )
    }
}

with open(
    REPORT_DIR / "model_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)


# --------------------------------------------------
# 17. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("TOPIC MODELLING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nLDA")
print("Topics generated:", len(lda_topics))
print("Perplexity:", round(lda_perplexity, 2))

print("\nNMF")
print("Topics generated:", len(nmf_topics))
print("Reconstruction error:", round(nmf_reconstruction_error, 4))

print("\nGenerated files:")
print("LDA topics:", LDA_DIR / "lda_topics.csv")
print("NMF topics:", NMF_DIR / "nmf_topics.csv")
print("Metrics:", REPORT_DIR / "model_metrics.json")

print("\n" + "=" * 70)
