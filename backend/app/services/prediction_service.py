
from app.services.preprocessing import preprocess_document


def predict_topics(text, models):

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text cannot be empty.")

    processed = preprocess_document(text)

    cleaned_text = processed["cleaned_text"]
    processed_text = processed["processed_text"]

    if not cleaned_text:
        raise ValueError("No meaningful text remains after cleaning.")

    if not processed_text:
        raise ValueError("No meaningful words remain after preprocessing.")

    results = {}

    # --------------------------------------------------
    # 1. LDA PREDICTION
    # --------------------------------------------------

    lda_model = models["lda"]["model"]
    lda_vectorizer = models["lda"]["vectorizer"]

    lda_matrix = lda_vectorizer.transform([processed_text])

    lda_distribution = lda_model.transform(lda_matrix)[0]

    lda_topic = int(lda_distribution.argmax())

    results["lda"] = {
        "topic_id": lda_topic,
        "confidence": round(
            float(lda_distribution[lda_topic]), 4
        ),
        "topic_distribution": [
            round(float(value), 4)
            for value in lda_distribution
        ]
    }

    # --------------------------------------------------
    # 2. NMF PREDICTION
    # --------------------------------------------------

    nmf_model = models["nmf"]["model"]
    nmf_vectorizer = models["nmf"]["vectorizer"]

    nmf_matrix = nmf_vectorizer.transform([processed_text])

    nmf_distribution = nmf_model.transform(nmf_matrix)[0]

    nmf_total = float(nmf_distribution.sum())

    if nmf_total > 0:
        nmf_distribution = nmf_distribution / nmf_total

    nmf_topic = int(nmf_distribution.argmax())

    results["nmf"] = {
        "topic_id": nmf_topic,
        "confidence": round(
            float(nmf_distribution[nmf_topic]), 4
        ),
        "topic_distribution": [
            round(float(value), 4)
            for value in nmf_distribution
        ]
    }

    # --------------------------------------------------
    # 3. BERTOPIC PREDICTION
    # --------------------------------------------------

    bertopic_model = models["bertopic"]["model"]
    embedding_model = models["bertopic"]["embedding_model"]

    embeddings = embedding_model.encode(
        [cleaned_text],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    bertopic_topics, bertopic_probabilities = (
        bertopic_model.transform(
            [cleaned_text],
            embeddings=embeddings
        )
    )

    bertopic_topic = int(bertopic_topics[0])

    if bertopic_topic == -1:
        bertopic_name = "Outlier / Unassigned"
        bertopic_words = []
        bertopic_confidence = None

    else:
        topic_info = bertopic_model.get_topic(
            bertopic_topic
        )

        bertopic_name = f"Topic {bertopic_topic}"

        bertopic_words = [
            word
            for word, score in topic_info[:10]
        ] if topic_info else []

        bertopic_confidence = None

        if bertopic_probabilities is not None:
            probability = bertopic_probabilities[0]

            if getattr(probability, "ndim", 0) > 0:
                bertopic_confidence = round(
                    float(probability.max()), 4
                )
            else:
                bertopic_confidence = round(
                    float(probability), 4
                )

    results["bertopic"] = {
        "topic_id": bertopic_topic,
        "topic_name": bertopic_name,
        "top_words": bertopic_words,
        "confidence": bertopic_confidence
    }

    return {
        "predictions": results
    }
