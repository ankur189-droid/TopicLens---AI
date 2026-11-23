
from sqlalchemy import Column, Integer, Text, DateTime, JSON, func

from app.database.base import TopicLensBase


class PredictionHistory(TopicLensBase):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)

    input_text = Column(Text, nullable=False)

    lda_prediction = Column(JSON, nullable=False)
    nmf_prediction = Column(JSON, nullable=False)
    bertopic_prediction = Column(JSON, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
