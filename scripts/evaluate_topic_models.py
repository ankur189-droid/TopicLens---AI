import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score
)


# ==========================================================
# 1. PATH CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "20_newsgroups_preprocessed.csv"
)

LDA_TOPICS_PATH = BASE_DIR / "models" / "lda" / "lda_topics.csv"
NMF_TOPICS_PATH = BASE_DIR / "models" / "nmf" / "nmf_topics.csv"

BERTOPIC_TOPICS_PATH = (
    BASE_DIR / "experiments" / "phase4" / "bertopic_topics.csv"
)

LDA_ASSIGNMENTS_PATH = (
    BASE_DIR / "experiments" / "phase3" / "lda_document_topics.csv"
)

NMF_ASSIGNMENTS_PATH = (
    BASE_DIR / "experiments" / "phase3" / "nmf_document_topics.csv"
)

BERTOPIC_ASSIGNMENTS_PATH = (
    BASE_DIR / "experiments" / "phase4" / "bertopic_document_topics.csv"
)

LDA_METRICS_PATH = (
    BASE_DIR / "reports" / "phase3" / "model_metrics.json"
)

BERTOPIC_METRICS_PATH = (
    BASE_DIR / "reports" / "phase4" / "bertopic_metrics.json"
)

REPORT_DIR = BASE_DIR / "reports" / "phase5"
EXPERIMENT_DIR = BASE_DIR / "experiments" / "phase5"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. CONFIGURATION
# ==========================================================

TOP_N_WORDS = 15

print("=" * 70)
print("TOPICLENS AI - TOPIC MODEL EVALUATION")
print("=" * 70)


# ==========================================================
# 3. LOAD DATASET
# ==========================================================

print("\nLoading preprocessed dataset...")

df = pd.read_csv(DATASET_PATH)

required_columns = [
    "document_id",
    "category",
    "cleaned_text",
    "processed_text"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required dataset columns: {missing_columns}"
    )

df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
df["processed_text"] = df["processed_text"].fillna("").astype(str)

print("Total original documents:", len(df))


# ==========================================================
# 4. TEXT TOKENIZATION
# ==========================================================

def tokenize_text(text):
    """
    Convert text into lowercase alphabetic tokens.
    """

    return [
        word.lower()
        for word in str(text).split()
        if word.isalpha() and len(word) > 1
    ]


# Use cleaned_text for all three models so that
# coherence is evaluated against the same corpus.

common_texts = [
    tokenize_text(text)
    for text in df["cleaned_text"]
]

common_dictionary = Dictionary(common_texts)

print("Common evaluation dictionary size:", len(common_dictionary))


# ==========================================================
# 5. HELPER: LOAD TOPIC WORDS FROM CSV
# ==========================================================

def load_simple_topics(file_path):
    """
    Load LDA or NMF topics from a CSV containing
    topic_id and top_words columns.
    """

    topic_df = pd.read_csv(file_path)

    required = {"topic_id", "top_words"}

    if not required.issubset(topic_df.columns):
        raise ValueError(
            f"{file_path} must contain columns: {required}"
        )

    topics = []

    for _, row in topic_df.iterrows():

        words = [
            word.strip().lower()
            for word in str(row["top_words"]).split(",")
            if word.strip()
        ]

        topics.append(words[:TOP_N_WORDS])

    return topics


# ==========================================================
# 6. HELPER: LOAD BERTOPIC TOPICS
# ==========================================================

def load_bertopic_topics(file_path):
    """
    Read BERTopic's get_topic_info() CSV output.

    The Representation column normally contains a
    serialized list of representative topic words.
    """

    topic_df = pd.read_csv(file_path)

    if "Topic" not in topic_df.columns:
        raise ValueError(
            "BERTopic topic CSV does not contain 'Topic'."
        )

    if "Representation" not in topic_df.columns:
        raise ValueError(
            "BERTopic topic CSV does not contain 'Representation'."
        )

    topics = []

    for _, row in topic_df.iterrows():

        topic_id = int(row["Topic"])

        # Exclude the HDBSCAN outlier topic.
        if topic_id == -1:
            continue

        representation = row["Representation"]

        if isinstance(representation, str):
            try:
                words = ast.literal_eval(representation)
            except (ValueError, SyntaxError):
                words = representation.strip("[]").split(",")
        else:
            words = []

        words = [
            str(word).strip().strip("'\"").lower()
            for word in words
            if str(word).strip()
        ]

        topics.append(words[:TOP_N_WORDS])

    return topics


# ==========================================================
# 7. LOAD ALL THREE MODELS' TOPIC WORDS
# ==========================================================

print("\nLoading discovered topics...")

lda_topics = load_simple_topics(LDA_TOPICS_PATH)
nmf_topics = load_simple_topics(NMF_TOPICS_PATH)
bertopic_topics = load_bertopic_topics(BERTOPIC_TOPICS_PATH)

print("LDA topics:", len(lda_topics))
print("NMF topics:", len(nmf_topics))
print("BERTopic topics:", len(bertopic_topics))


# ==========================================================
# 8. TOPIC COHERENCE
# ==========================================================

def calculate_coherence(topics, texts, dictionary, model_name):
    """
    Calculate C_v coherence for a collection of topics.
    """

    valid_topics = []

    for topic in topics:

        # Keep only words present in the evaluation dictionary.
        filtered_words = [
            word for word in topic
            if word in dictionary.token2id
        ]

        # A topic needs at least two known words.
        if len(filtered_words) >= 2:
            valid_topics.append(filtered_words)

    if not valid_topics:
        print(f"Warning: No valid topics for {model_name}.")
        return np.nan, []

    coherence_model = CoherenceModel(
        topics=valid_topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
        topn=TOP_N_WORDS,
        processes=1
    )

    coherence_score = coherence_model.get_coherence()

    per_topic_scores = coherence_model.get_coherence_per_topic()

    print(
        f"{model_name} C_v coherence:",
        round(coherence_score, 4)
    )

    return coherence_score, per_topic_scores


print("\n" + "=" * 70)
print("CALCULATING TOPIC COHERENCE")
print("=" * 70)

lda_coherence, lda_topic_coherence = calculate_coherence(
    lda_topics,
    common_texts,
    common_dictionary,
    "LDA"
)

nmf_coherence, nmf_topic_coherence = calculate_coherence(
    nmf_topics,
    common_texts,
    common_dictionary,
    "NMF"
)

bertopic_coherence, bertopic_topic_coherence = calculate_coherence(
    bertopic_topics,
    common_texts,
    common_dictionary,
    "BERTopic"
)


# ==========================================================
# 9. TOPIC DIVERSITY
# ==========================================================

def calculate_topic_diversity(topics):
    """
    Topic diversity = unique top words across topics /
                      total top words across topics.
    """

    all_words = [
        word
        for topic in topics
        for word in topic[:TOP_N_WORDS]
    ]

    if not all_words:
        return 0.0

    return len(set(all_words)) / len(all_words)


lda_diversity = calculate_topic_diversity(lda_topics)
nmf_diversity = calculate_topic_diversity(nmf_topics)
bertopic_diversity = calculate_topic_diversity(bertopic_topics)

print("\n" + "=" * 70)
print("TOPIC DIVERSITY")
print("=" * 70)

print("LDA:", round(lda_diversity, 4))
print("NMF:", round(nmf_diversity, 4))
print("BERTopic:", round(bertopic_diversity, 4))


# ==========================================================
# 10. LOAD DOCUMENT ASSIGNMENTS
# ==========================================================

print("\nLoading document-topic assignments...")

lda_assignments = pd.read_csv(LDA_ASSIGNMENTS_PATH)
nmf_assignments = pd.read_csv(NMF_ASSIGNMENTS_PATH)
bertopic_assignments = pd.read_csv(BERTOPIC_ASSIGNMENTS_PATH)


# ==========================================================
# 11. EXTERNAL EVALUATION
# ==========================================================

def evaluate_assignments(file_df, model_name, common_ids):
    """
    Evaluate every model on the same document IDs.

    For BERTopic:
    - Full metrics include outlier topic -1.
    - Inlier metrics exclude outliers.
    """

    required = {
        "document_id",
        "category",
        "dominant_topic"
    }

    if not required.issubset(file_df.columns):
        raise ValueError(
            f"{model_name} assignment CSV must contain {required}"
        )

    merged = file_df.merge(
        df[["document_id", "category"]],
        on="document_id",
        how="inner",
        suffixes=("_assignment", "_original"),
        validate="one_to_one"
    )

    if not (
        merged["category_assignment"]
        == merged["category_original"]
    ).all():
        raise ValueError(
            f"Category mismatch in {model_name}."
        )

    merged = merged[
        merged["document_id"].isin(common_ids)
    ].copy()

    outliers = int(
        (merged["dominant_topic"] == -1).sum()
    )

    # Evaluate all documents, including BERTopic outliers.
    full_nmi = normalized_mutual_info_score(
        merged["category_original"],
        merged["dominant_topic"]
    )

    full_ari = adjusted_rand_score(
        merged["category_original"],
        merged["dominant_topic"]
    )

    # Evaluate only documents assigned to actual topics.
    inliers = merged[
        merged["dominant_topic"] != -1
    ]

    if len(inliers) > 0:

        inlier_nmi = normalized_mutual_info_score(
            inliers["category_original"],
            inliers["dominant_topic"]
        )

        inlier_ari = adjusted_rand_score(
            inliers["category_original"],
            inliers["dominant_topic"]
        )

    else:
        inlier_nmi = np.nan
        inlier_ari = np.nan

    coverage = (
        len(inliers) / len(merged) * 100
        if len(merged) else 0
    )

    return {
        "documents_evaluated": int(len(merged)),
        "outlier_documents": outliers,
        "topic_coverage_percentage": round(coverage, 2),
        "nmi_full": round(float(full_nmi), 4),
        "ari_full": round(float(full_ari), 4),
        "nmi_inliers_only": (
            round(float(inlier_nmi), 4)
            if np.isfinite(inlier_nmi) else None
        ),
        "ari_inliers_only": (
            round(float(inlier_ari), 4)
            if np.isfinite(inlier_ari) else None
        )
    }


print("\n" + "=" * 70)
print("EXTERNAL EVALUATION AGAINST ORIGINAL CATEGORIES")
print("=" * 70)

# Find documents present in all three assignment files.
common_ids = set(lda_assignments["document_id"])
common_ids &= set(nmf_assignments["document_id"])
common_ids &= set(bertopic_assignments["document_id"])

print("\nCommon documents across all models:", len(common_ids))

if len(common_ids) == 0:
    raise ValueError("No common document IDs found.")

lda_external = evaluate_assignments(
    lda_assignments, "LDA", common_ids
)

nmf_external = evaluate_assignments(
    nmf_assignments, "NMF", common_ids
)

bertopic_external = evaluate_assignments(
    bertopic_assignments, "BERTopic", common_ids
)

print("\nLDA:", lda_external)
print("NMF:", nmf_external)
print("BERTopic:", bertopic_external)


# ==========================================================
# 12. LOAD TRAINING METRICS
# ==========================================================

with open(LDA_METRICS_PATH, "r", encoding="utf-8") as file:
    phase3_metrics = json.load(file)

with open(BERTOPIC_METRICS_PATH, "r", encoding="utf-8") as file:
    phase4_metrics = json.load(file)

lda_time = phase3_metrics["lda"]["training_time_seconds"]
nmf_time = phase3_metrics["nmf"]["training_time_seconds"]

bertopic_time = phase4_metrics[
    "topic_model_training_time_seconds"
]

lda_perplexity = phase3_metrics["lda"]["perplexity"]
nmf_reconstruction_error = phase3_metrics["nmf"][
    "reconstruction_error"
]


# ==========================================================
# 13. CREATE MODEL COMPARISON TABLE
# ==========================================================

comparison = pd.DataFrame([
    {
        "model": "LDA",
        "number_of_topics": len(lda_topics),
        "coherence_cv": lda_coherence,
        "topic_diversity": lda_diversity,
        "nmi": lda_external["nmi_full"],
        "ari": lda_external["ari_full"],
        "nmi_inliers_only": lda_external["nmi_inliers_only"],
        "ari_inliers_only": lda_external["ari_inliers_only"],
        "topic_coverage_percentage":
            lda_external["topic_coverage_percentage"],
        "outlier_documents": lda_external["outlier_documents"],
        "training_time_seconds": lda_time,
        "perplexity": lda_perplexity,
        "reconstruction_error": np.nan
    },
    {
        "model": "NMF",
        "number_of_topics": len(nmf_topics),
        "coherence_cv": nmf_coherence,
        "topic_diversity": nmf_diversity,
        "nmi": nmf_external["nmi_full"],
        "ari": nmf_external["ari_full"],
        "nmi_inliers_only": nmf_external["nmi_inliers_only"],
        "ari_inliers_only": nmf_external["ari_inliers_only"],
        "topic_coverage_percentage":
            nmf_external["topic_coverage_percentage"],
        "outlier_documents": nmf_external["outlier_documents"],
        "training_time_seconds": nmf_time,
        "perplexity": np.nan,
        "reconstruction_error": nmf_reconstruction_error
    },
    {
        "model": "BERTopic",
        "number_of_topics": len(bertopic_topics),
        "coherence_cv": bertopic_coherence,
        "topic_diversity": bertopic_diversity,
        "nmi": bertopic_external["nmi_full"],
        "ari": bertopic_external["ari_full"],
        "nmi_inliers_only": bertopic_external["nmi_inliers_only"],
        "ari_inliers_only": bertopic_external["ari_inliers_only"],
        "topic_coverage_percentage":
            bertopic_external["topic_coverage_percentage"],
        "outlier_documents": bertopic_external["outlier_documents"],
        "training_time_seconds": bertopic_time,
        "perplexity": np.nan,
        "reconstruction_error": np.nan
    }
])

comparison.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(comparison.to_string(index=False))


# ==========================================================
# 14. SAVE PER-TOPIC COHERENCE
# ==========================================================

topic_coherence_rows = []

for model_name, scores in [
    ("LDA", lda_topic_coherence),
    ("NMF", nmf_topic_coherence),
    ("BERTopic", bertopic_topic_coherence)
]:

    for topic_id, score in enumerate(scores):

        topic_coherence_rows.append({
            "model": model_name,
            "topic_id": topic_id,
            "coherence_cv": float(score)
        })

pd.DataFrame(topic_coherence_rows).to_csv(
    REPORT_DIR / "topic_coherence.csv",
    index=False
)


# ==========================================================
# 15. SAVE TOPIC DIVERSITY
# ==========================================================

diversity_df = pd.DataFrame([
    {"model": "LDA", "topic_diversity": lda_diversity},
    {"model": "NMF", "topic_diversity": nmf_diversity},
    {"model": "BERTopic", "topic_diversity": bertopic_diversity}
])

diversity_df.to_csv(
    REPORT_DIR / "topic_diversity.csv",
    index=False
)


# ==========================================================
# 16. SAVE EVALUATION SUMMARY
# ==========================================================

summary = {
    "dataset": "20 Newsgroups",

    "evaluation_methods": [
        "C_v topic coherence",
        "Topic diversity",
        "Normalized Mutual Information",
        "Adjusted Rand Index",
        "Topic coverage"
    ],

    "coherence_corpus": {
        "lda": "cleaned_text",
        "nmf": "cleaned_text",
        "bertopic": "cleaned_text"
    },

    "top_words_per_topic": TOP_N_WORDS,

    "models": {
        "LDA": {
            "coherence_cv": (
                float(lda_coherence)
                if np.isfinite(lda_coherence) else None
            ),
            "topic_diversity": lda_diversity,
            **lda_external,
            "training_time_seconds": lda_time,
            "perplexity": lda_perplexity
        },

        "NMF": {
            "coherence_cv": (
                float(nmf_coherence)
                if np.isfinite(nmf_coherence) else None
            ),
            "topic_diversity": nmf_diversity,
            **nmf_external,
            "training_time_seconds": nmf_time,
            "reconstruction_error": nmf_reconstruction_error
        },

        "BERTopic": {
            "coherence_cv": (
                float(bertopic_coherence)
                if np.isfinite(bertopic_coherence) else None
            ),
            "topic_diversity": bertopic_diversity,
            **bertopic_external,
            "training_time_seconds": bertopic_time
        }
    }
}

with open(
    REPORT_DIR / "evaluation_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)


# ==========================================================
# 17. FINAL OUTPUT
# ==========================================================

print("\n" + "=" * 70)
print("PHASE 5 MODEL EVALUATION COMPLETED")
print("=" * 70)

print("\nReports saved in:", REPORT_DIR)

print("\nGenerated files:")
print("1. model_comparison.csv")
print("2. topic_coherence.csv")
print("3. topic_diversity.csv")
print("4. evaluation_summary.json")

print("\n" + "=" * 70)