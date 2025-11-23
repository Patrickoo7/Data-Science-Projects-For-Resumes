"""Analytics API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from ..core.database import get_db
from ..core.security import get_current_active_user
from ..models.database import User, Prediction, UsageStats

router = APIRouter()


@router.get("/usage/summary")
async def get_usage_summary(
    days: int = 30,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get usage summary for the last N days."""
    since_date = datetime.utcnow() - timedelta(days=days)

    total_predictions = db.query(func.count(Prediction.id)).filter(
        Prediction.user_id == current_user.id,
        Prediction.created_at >= since_date
    ).scalar()

    sentiment_breakdown = db.query(
        Prediction.sentiment,
        func.count(Prediction.id)
    ).filter(
        Prediction.user_id == current_user.id,
        Prediction.created_at >= since_date
    ).group_by(Prediction.sentiment).all()

    avg_confidence = db.query(func.avg(Prediction.confidence)).filter(
        Prediction.user_id == current_user.id,
        Prediction.created_at >= since_date
    ).scalar()

    return {
        "period_days": days,
        "total_predictions": total_predictions,
        "sentiment_breakdown": {s: c for s, c in sentiment_breakdown},
        "average_confidence": round(float(avg_confidence or 0), 4),
        "total_characters_processed": db.query(
            func.sum(func.length(Prediction.text))
        ).filter(
            Prediction.user_id == current_user.id,
            Prediction.created_at >= since_date
        ).scalar() or 0
    }


@router.get("/usage/daily")
async def get_daily_usage(
    days: int = 30,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get daily usage statistics."""
    since_date = datetime.utcnow() - timedelta(days=days)

    stats = db.query(UsageStats).filter(
        UsageStats.user_id == current_user.id,
        UsageStats.date >= since_date
    ).order_by(UsageStats.date).all()

    return [{
        "date": s.date.isoformat(),
        "total_requests": s.total_requests,
        "successful_requests": s.successful_requests,
        "failed_requests": s.failed_requests,
        "avg_processing_time_ms": s.avg_processing_time_ms
    } for s in stats]


@router.get("/sentiment/trends")
async def get_sentiment_trends(
    days: int = 30,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get sentiment trends over time."""
    since_date = datetime.utcnow() - timedelta(days=days)

    predictions = db.query(
        func.date(Prediction.created_at).label('date'),
        Prediction.sentiment,
        func.count(Prediction.id).label('count')
    ).filter(
        Prediction.user_id == current_user.id,
        Prediction.created_at >= since_date
    ).group_by(
        func.date(Prediction.created_at),
        Prediction.sentiment
    ).all()

    return [{"date": p.date.isoformat(), "sentiment": p.sentiment, "count": p.count} for p in predictions]
