
from pydantic import BaseModel
from typing import Optional, List


class TopicItem(BaseModel):
    topic_id: int
    name: str
    keywords: List[str]
    document_count: Optional[int] = None
    representative_docs: Optional[List[str]] = None
    is_outlier: bool = False


class TopicListResponse(BaseModel):
    model: str
    total_topics: int
    topics: List[TopicItem]


class TopicDetailResponse(BaseModel):
    model: str
    topic: TopicItem
