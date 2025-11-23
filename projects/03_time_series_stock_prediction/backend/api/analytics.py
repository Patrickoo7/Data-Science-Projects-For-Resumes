"""Analytics API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from ..core.database import get_db
from ..core.security import get_current_active_user
from ..models.database import User, Prediction

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

    ticker_breakdown = db.query(
        Prediction.ticker,
        func.count(Prediction.id)
    ).filter(
        Prediction.user_id == current_user.id,
        Prediction.created_at >= since_date
    ).group_by(Prediction.ticker).all()

    avg_processing_time = db.query(func.avg(Prediction.processing_time_ms)).filter(
        Prediction.user_id == current_user.id,
        Prediction.created_at >= since_date
    ).scalar()

    return {
        "period_days": days,
        "total_predictions": total_predictions,
        "ticker_breakdown": {t: c for t, c in ticker_breakdown},
        "average_processing_time_ms": round(float(avg_processing_time or 0), 2),
        "most_predicted_ticker": ticker_breakdown[0][0] if ticker_breakdown else None
    }


@router.get("/predictions/trends")
async def get_prediction_trends(
    days: int = 30,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get prediction trends over time."""
    since_date = datetime.utcnow() - timedelta(days=days)

    predictions = db.query(
        func.date(Prediction.created_at).label('date'),
        func.count(Prediction.id).label('count')
    ).filter(
        Prediction.user_id == current_user.id,
        Prediction.created_at >= since_date
    ).group_by(
        func.date(Prediction.created_at)
    ).all()

    return [{"date": p.date.isoformat(), "count": p.count} for p in predictions]
