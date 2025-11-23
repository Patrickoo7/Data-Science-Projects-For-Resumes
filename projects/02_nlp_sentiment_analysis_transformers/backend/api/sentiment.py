"""
Sentiment analysis API endpoints.
Main prediction endpoints with rate limiting and usage tracking.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime

from ..core.database import get_db
from ..core.security import get_current_active_user
from ..models.database import User, Prediction, UsageStats
from ..services.sentiment_service import analyzer

router = APIRouter()


# Pydantic models
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    use_cache: bool = Field(True, description="Use cached results if available")


class SentimentBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    use_cache: bool = Field(True, description="Use cached results if available")


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
    processing_time_ms: float
    prediction_id: str = None


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_count: int
    avg_processing_time_ms: float


# Background task to save prediction
async def save_prediction_task(
    user_id: int,
    result: dict,
    db: Session
):
    """Save prediction to database in background."""
    try:
        prediction = Prediction(
            user_id=user_id,
            text=result['text'],
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            model_version=result['model_version'],
            processing_time_ms=result['processing_time_ms']
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)

        # Update usage stats
        today = datetime.utcnow().date()
        stats = db.query(UsageStats).filter(
            UsageStats.user_id == user_id,
            UsageStats.date >= today
        ).first()

        if stats:
            stats.total_requests += 1
            stats.successful_requests += 1
            stats.total_characters_processed += len(result['text'])
            # Update average processing time
            n = stats.successful_requests
            stats.avg_processing_time_ms = (
                (stats.avg_processing_time_ms * (n - 1) + result['processing_time_ms']) / n
            )
        else:
            stats = UsageStats(
                user_id=user_id,
                date=datetime.utcnow(),
                total_requests=1,
                successful_requests=1,
                total_characters_processed=len(result['text']),
                avg_processing_time_ms=result['processing_time_ms']
            )
            db.add(stats)

        db.commit()

    except Exception as e:
        print(f"Error saving prediction: {e}")
        db.rollback()


@router.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(
    request: SentimentRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Analyze sentiment of a single text.

    - **text**: The text to analyze (1-10000 characters)
    - **use_cache**: Whether to use cached results (default: true)

    Returns sentiment (positive/negative/neutral) with confidence scores.
    """

    try:
        # Perform prediction
        result = await analyzer.predict(request.text, use_cache=request.use_cache)

        # Save prediction in background
        background_tasks.add_task(save_prediction_task, current_user.id, result, db)

        return result

    except Exception as e:
        # Update failed request stats
        today = datetime.utcnow().date()
        stats = db.query(UsageStats).filter(
            UsageStats.user_id == current_user.id,
            UsageStats.date >= today
        ).first()

        if stats:
            stats.total_requests += 1
            stats.failed_requests += 1
            db.commit()

        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_batch_sentiment(
    request: SentimentBatchRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Analyze sentiment of multiple texts in batch.

    - **texts**: List of texts to analyze (max 100 texts)
    - **use_cache**: Whether to use cached results (default: true)

    Returns list of sentiment predictions.
    """

    try:
        # Perform batch prediction
        results = await analyzer.predict_batch(request.texts, use_cache=request.use_cache)

        # Calculate average processing time
        avg_time = sum(r['processing_time_ms'] for r in results) / len(results)

        # Save predictions in background
        for result in results:
            background_tasks.add_task(save_prediction_task, current_user.id, result, db)

        return {
            "results": results,
            "total_count": len(results),
            "avg_processing_time_ms": round(avg_time, 2)
        }

    except Exception as e:
        # Update failed request stats
        today = datetime.utcnow().date()
        stats = db.query(UsageStats).filter(
            UsageStats.user_id == current_user.id,
            UsageStats.date >= today
        ).first()

        if stats:
            stats.total_requests += len(request.texts)
            stats.failed_requests += len(request.texts)
            db.commit()

        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/model/info")
async def get_model_info(current_user: User = Depends(get_current_active_user)):
    """
    Get information about the sentiment analysis model.

    Returns model architecture, parameters, and configuration details.
    """
    return analyzer.get_model_info()


@router.get("/predictions/recent")
async def get_recent_predictions(
    limit: int = 10,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get recent predictions for current user.

    - **limit**: Number of predictions to return (default: 10, max: 100)
    """
    if limit > 100:
        limit = 100

    predictions = db.query(Prediction).filter(
        Prediction.user_id == current_user.id
    ).order_by(Prediction.created_at.desc()).limit(limit).all()

    return predictions


@router.get("/predictions/{prediction_id}")
async def get_prediction_by_id(
    prediction_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific prediction by ID."""
    prediction = db.query(Prediction).filter(
        Prediction.uuid == prediction_id,
        Prediction.user_id == current_user.id
    ).first()

    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return prediction


@router.delete("/predictions/{prediction_id}")
async def delete_prediction(
    prediction_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a prediction."""
    prediction = db.query(Prediction).filter(
        Prediction.uuid == prediction_id,
        Prediction.user_id == current_user.id
    ).first()

    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    db.delete(prediction)
    db.commit()

    return {"message": "Prediction deleted successfully"}
