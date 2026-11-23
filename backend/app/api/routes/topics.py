import ast
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.schemas.topics import (
    TopicItem,
    TopicListResponse,
    TopicDetailResponse,
)

router = APIRouter(
    prefix="/api/v1/topics",
    tags=["Topics"]
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TOPIC_FILES = {
    "lda": PROJECT_ROOT / "models" / "lda" / "lda_topics.csv",
    "nmf": PROJECT_ROOT / "models" / "nmf" / "nmf_topics.csv",
    "bertopic": PROJECT_ROOT / "experiments" / "phase4" / "bertopic_topics.csv",
}


def parse_list(value):
    if isinstance(value, list):
        return [str(item) for item in value]

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (ValueError, SyntaxError):
        pass

    return [word.strip() for word in str(value).split(",") if word.strip()]


def load_topics(model: str):
    file_path = TOPIC_FILES.get(model)

    if file_path is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid model. Choose lda, nmf, or bertopic.",
        )

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Topic file not found: {file_path}",
        )

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read topic file: {str(exc)}",
        )

    topics = []

    for _, row in df.iterrows():
        if model in ("lda", "nmf"):
            topic_id = int(row["topic_id"])
            topic_name = f"Topic {topic_id}"
            keywords = parse_list(row["top_words"])
            document_count = None
            representative_docs = None
            is_outlier = False

        else:
            topic_id = int(row["Topic"])
            topic_name = str(row["Name"])
            keywords = parse_list(row["Representation"])

            document_count = int(row["Count"])

            representative_docs = parse_list(
                row["Representative_Docs"]
            )

            is_outlier = topic_id == -1

        topics.append(
            TopicItem(
                topic_id=topic_id,
                name=topic_name,
                keywords=keywords,
                document_count=document_count,
                representative_docs=representative_docs,
                is_outlier=is_outlier,
            )
        )

    return topics


@router.get("/", response_model=TopicListResponse)
def get_topics(
    model: str = Query("lda", pattern="^(lda|nmf|bertopic)$"),
    search: str = Query("", description="Search topic names or keywords"),
    include_outliers: bool = Query(False),
):
    topics = load_topics(model)

    if not include_outliers:
        topics = [topic for topic in topics if not topic.is_outlier]

    if search.strip():
        search_text = search.strip().lower()

        topics = [
            topic
            for topic in topics
            if search_text in topic.topic_name.lower()
            or any(search_text in word.lower() for word in topic.keywords)
        ]

    return TopicListResponse(
        model=model,
        total_topics=len(topics),
        topics=topics,
    )


@router.get("/{model}/{topic_id}", response_model=TopicDetailResponse)
def get_topic_detail(
    model: str,
    topic_id: int,
):
    topics = load_topics(model)

    for topic in topics:
        if topic.topic_id == topic_id:
            return TopicDetailResponse(
                model=model,
                topic=topic,
            )

    raise HTTPException(
        status_code=404,
        detail=f"Topic {topic_id} not found in {model}.",
    )