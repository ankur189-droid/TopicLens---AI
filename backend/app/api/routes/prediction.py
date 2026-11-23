from fastapi import APIRouter, Request, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import predict_topics
from app.database.session import get_topiclens_db
from app.models.prediction import PredictionHistory

router = APIRouter(prefix="/api/v1", tags=["Topic Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict_document(
    request: PredictionRequest,
    app_request: Request,
    db: Session = Depends(get_topiclens_db)
):
    try:
        models = app_request.app.state.models

        result = predict_topics(request.text, models)
        predictions = result["predictions"]

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during prediction."
        )

    try:
        history = PredictionHistory(
            input_text=request.text,
            lda_prediction=predictions["lda"],
            nmf_prediction=predictions["nmf"],
            bertopic_prediction=predictions["bertopic"]
        )

        db.add(history)
        db.commit()
        db.refresh(history)

        return {
            "status": "success",
            "history_id": history.id,
            "predictions": predictions
        }

    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Prediction was generated, but could not be saved to database."
        )


@router.get("/history")
def get_prediction_history(
    db: Session = Depends(get_topiclens_db),
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    try:
        records = (
            db.query(PredictionHistory)
            .order_by(PredictionHistory.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        total = db.query(PredictionHistory).count()

        return {
            "status": "success",
            "total_records": total,
            "limit": limit,
            "offset": offset,
            "history": [
                {
                    "id": record.id,
                    "input_text": record.input_text,
                    "lda_prediction": record.lda_prediction,
                    "nmf_prediction": record.nmf_prediction,
                    "bertopic_prediction": record.bertopic_prediction,
                    "created_at": record.created_at.isoformat()
                }
                for record in records
            ]
        }

    except SQLAlchemyError:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve prediction history."
        )


@router.get("/history/{history_id}")
def get_prediction_by_id(
    history_id: int,
    db: Session = Depends(get_topiclens_db)
):
    try:
        record = (
            db.query(PredictionHistory)
            .filter(PredictionHistory.id == history_id)
            .first()
        )

        if record is None:
            raise HTTPException(
                status_code=404,
                detail="Prediction history record not found."
            )

        return {
            "status": "success",
            "history": {
                "id": record.id,
                "input_text": record.input_text,
                "lda_prediction": record.lda_prediction,
                "nmf_prediction": record.nmf_prediction,
                "bertopic_prediction": record.bertopic_prediction,
                "created_at": record.created_at.isoformat()
            }
        }

    except SQLAlchemyError:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve prediction history."
        )