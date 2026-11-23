
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer

from umap import UMAP
from hdbscan import HDBSCAN


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

MODEL_DIR = BASE_DIR / "models" / "bertopic"
EXPERIMENT_DIR = BASE_DIR / "experiments" / "phase4"
REPORT_DIR = BASE_DIR / "reports" / "phase4"

EMBEDDINGS_DIR = BASE_DIR / "storage" / "embeddings"

for directory in [
    MODEL_DIR,
    EXPERIMENT_DIR,
    REPORT_DIR,
    EMBEDDINGS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 32
TOP_WORDS = 15

MIN_TOPIC_SIZE = 30
MIN_CLUSTER_SIZE = 30
MIN_SAMPLES = 10

RANDOM_STATE = 42

EMBEDDINGS_PATH = (
    EMBEDDINGS_DIR / "20_newsgroups_embeddings.npy"
)


# --------------------------------------------------
# 3. GPU CONFIGURATION
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("TOPICLENS AI - BERTOPIC TOPIC MODELLING")
print("=" * 70)

print("\nSelected device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU unavailable. Using CPU for embeddings.")


# --------------------------------------------------
# 4. LOAD DATASET
# --------------------------------------------------

print("\nLoading preprocessed dataset...")

df = pd.read_csv(DATASET_PATH)

df["cleaned_text"] = (
    df["cleaned_text"].fillna("").astype(str)
)

# Keep documents that contain usable cleaned text.
df = df[
    df["cleaned_text"].str.strip().ne("")
].copy()

df.reset_index(drop=True, inplace=True)

documents = df["cleaned_text"].tolist()

print("Original records:", 18846)
print("Documents available:", len(documents))
print("Documents excluded:", 18846 - len(documents))


# --------------------------------------------------
# 5. LOAD SENTENCE TRANSFORMER
# --------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1: LOADING SENTENCE TRANSFORMER")
print("=" * 70)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device=device
)

# Limit the input sequence length to control memory
# and processing time for long Usenet documents.
embedding_model.max_seq_length = 256

print("Embedding model:", EMBEDDING_MODEL_NAME)
print("Embedding device:", device)
print("Maximum sequence length:", embedding_model.max_seq_length)


# --------------------------------------------------
# 6. GENERATE DOCUMENT EMBEDDINGS
# --------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2: GENERATING DOCUMENT EMBEDDINGS")
print("=" * 70)

embedding_start = time.time()

if EMBEDDINGS_PATH.exists():

    print("Existing embeddings found.")
    print("Loading saved embeddings...")

    embeddings = np.load(EMBEDDINGS_PATH)

    if embeddings.shape[0] != len(documents):
        raise ValueError(
            "Saved embeddings do not match the current dataset. "
            "Delete the old embedding file and run again."
        )

else:

    print("Generating embeddings. Please wait...")

    embeddings = embedding_model.encode(
        documents,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    np.save(EMBEDDINGS_PATH, embeddings)

    print("Embeddings saved:", EMBEDDINGS_PATH)

embedding_time = time.time() - embedding_start

print("Embedding shape:", embeddings.shape)
print("Embedding time:", round(embedding_time, 2), "seconds")


# --------------------------------------------------
# 7. CONFIGURE UMAP
# --------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3: CONFIGURING UMAP")
print("=" * 70)

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=RANDOM_STATE,
    low_memory=True
)

print("UMAP configuration completed.")


# --------------------------------------------------
# 8. CONFIGURE HDBSCAN
# --------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4: CONFIGURING HDBSCAN")
print("=" * 70)

hdbscan_model = HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

print("HDBSCAN configuration completed.")


# --------------------------------------------------
# 9. CONFIGURE COUNT VECTORIZER
# --------------------------------------------------

vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
)


# --------------------------------------------------
# 10. INITIALIZE BERTOPIC
# --------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5: INITIALIZING BERTOPIC")
print("=" * 70)

topic_model = BERTopic(
    embedding_model=None,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    min_topic_size=MIN_TOPIC_SIZE,
    top_n_words=TOP_WORDS,
    calculate_probabilities=False,
    verbose=True
)

print("BERTopic initialized successfully.")


# --------------------------------------------------
# 11. TRAIN BERTOPIC
# --------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6: TRAINING BERTOPIC MODEL")
print("=" * 70)

training_start = time.time()

topics, probabilities = topic_model.fit_transform(
    documents,
    embeddings
)

training_time = time.time() - training_start

print("\nBERTopic training completed.")
print("Training time:", round(training_time, 2), "seconds")


# --------------------------------------------------
# 12. EXTRACT TOPIC INFORMATION
# --------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7: EXTRACTING DISCOVERED TOPICS")
print("=" * 70)

topic_info = topic_model.get_topic_info()

topic_info.to_csv(
    EXPERIMENT_DIR / "bertopic_topics.csv",
    index=False
)

print("\nDISCOVERED TOPICS:\n")

for topic_id in topic_model.get_topics():

    if topic_id == -1:
        continue

    topic_words = topic_model.get_topic(topic_id)

    if topic_words:
        words = [
            word for word, score in topic_words[:TOP_WORDS]
        ]

        print(f"\nTopic {topic_id}:")
        print(", ".join(words))


# --------------------------------------------------
# 13. SAVE DOCUMENT TOPIC ASSIGNMENTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8: SAVING DOCUMENT ASSIGNMENTS")
print("=" * 70)

assignments = pd.DataFrame({
    "document_id": df["document_id"].values,
    "category": df["category"].values,
    "dominant_topic": topics
})

assignments.to_csv(
    EXPERIMENT_DIR / "bertopic_document_topics.csv",
    index=False
)

print(
    "Document assignments saved:",
    EXPERIMENT_DIR / "bertopic_document_topics.csv"
)


# --------------------------------------------------
# 14. CALCULATE TOPIC STATISTICS
# --------------------------------------------------

total_documents = len(topics)

outlier_count = int(np.sum(np.array(topics) == -1))

discovered_topic_count = len(
    set(topics) - {-1}
)

assigned_count = total_documents - outlier_count

outlier_percentage = (
    (outlier_count / total_documents) * 100
    if total_documents else 0
)

report = {
    "dataset": "20 Newsgroups",
    "embedding_model": EMBEDDING_MODEL_NAME,
    "embedding_device": device,
    "embedding_dimension": int(embeddings.shape[1]),
    "documents_used": int(total_documents),
    "documents_excluded": int(18846 - total_documents),

    "number_of_discovered_topics": int(
        discovered_topic_count
    ),

    "documents_assigned_to_topics": int(assigned_count),
    "outlier_documents": int(outlier_count),
    "outlier_percentage": round(outlier_percentage, 2),

    "embedding_generation_time_seconds": round(
        embedding_time, 2
    ),

    "topic_model_training_time_seconds": round(
        training_time, 2
    ),

    "configuration": {
        "batch_size": BATCH_SIZE,
        "max_sequence_length": 256,
        "umap_neighbors": 15,
        "umap_components": 5,
        "min_topic_size": MIN_TOPIC_SIZE,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES
    }
}

with open(
    REPORT_DIR / "bertopic_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(report, file, indent=4)


# --------------------------------------------------
# 15. SAVE TRAINED MODEL
# --------------------------------------------------

print("\n" + "=" * 70)
print("STEP 9: SAVING BERTOPIC MODEL")
print("=" * 70)

# Provide a direct file path inside your MODEL_DIR
model_file_path = MODEL_DIR / "bertopic_model"
topic_model.save(
    str(model_file_path),
    serialization="safetensors",  # Safetensors/default serialization is safer than raw pickle
    save_embedding_model=False    # Keeps the save lightweight since we use all-MiniLM-L6-v2 by name
)


# --------------------------------------------------
# 16. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("BERTOPIC TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("Documents used:", total_documents)
print("Topics discovered:", discovered_topic_count)
print("Documents assigned:", assigned_count)
print("Outlier documents:", outlier_count)
print("Outlier percentage:", round(outlier_percentage, 2), "%")

print("\nEmbedding time:", round(embedding_time, 2), "seconds")
print("Topic modelling time:", round(training_time, 2), "seconds")

print("\nGenerated files:")
print("Topic information:", EXPERIMENT_DIR / "bertopic_topics.csv")
print("Document assignments:", EXPERIMENT_DIR / "bertopic_document_topics.csv")
print("Metrics:", REPORT_DIR / "bertopic_metrics.json")
print("Embeddings:", EMBEDDINGS_PATH)
print("Model:", MODEL_DIR)

print("\n" + "=" * 70)
