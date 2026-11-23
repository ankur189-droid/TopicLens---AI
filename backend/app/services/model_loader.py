
from pathlib import Path

import joblib
import torch

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[3]

# Model directories
LDA_DIR = BASE_DIR / "models" / "lda"
NMF_DIR = BASE_DIR / "models" / "nmf"

BERTOPIC_DIR = (
    BASE_DIR / "models" / "bertopic" / "bertopic_model"
)

# Embedding model used during BERTopic training
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_models():

    print("=" * 60)
    print("TOPICLENS AI - LOADING TRAINED MODELS")
    print("=" * 60)

    models = {}

    # --------------------------------------------------
    # 1. LOAD LDA
    # --------------------------------------------------

    print("\nLoading LDA model...")

    lda_model = joblib.load(
        LDA_DIR / "lda_model.joblib"
    )

    count_vectorizer = joblib.load(
        LDA_DIR / "count_vectorizer.joblib"
    )

    models["lda"] = {
        "model": lda_model,
        "vectorizer": count_vectorizer
    }

    print("LDA model loaded successfully.")


    # --------------------------------------------------
    # 2. LOAD NMF
    # --------------------------------------------------

    print("\nLoading NMF model...")

    nmf_model = joblib.load(
        NMF_DIR / "nmf_model.joblib"
    )

    tfidf_vectorizer = joblib.load(
        NMF_DIR / "tfidf_vectorizer.joblib"
    )

    models["nmf"] = {
        "model": nmf_model,
        "vectorizer": tfidf_vectorizer
    }

    print("NMF model loaded successfully.")


    # --------------------------------------------------
    # 3. SELECT COMPUTATION DEVICE
    # --------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nSelected device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))


    # --------------------------------------------------
    # 4. LOAD SENTENCE TRANSFORMER
    # --------------------------------------------------

    print("\nLoading Sentence Transformer...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device=device
    )

    embedding_model.max_seq_length = 256

    print("Sentence Transformer loaded successfully.")


    # --------------------------------------------------
    # 5. LOAD BERTOPIC
    # --------------------------------------------------

    print("\nLoading BERTopic model...")

    bertopic_model = BERTopic.load(
        str(BERTOPIC_DIR),
        embedding_model=None
    )

    models["bertopic"] = {
        "model": bertopic_model,
        "embedding_model": embedding_model,
        "device": device
    }

    print("BERTopic model loaded successfully.")

    print("\n" + "=" * 60)
    print("ALL MODELS LOADED SUCCESSFULLY")
    print("=" * 60)

    return models
