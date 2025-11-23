"""Admin API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from ..core.database import get_db
from ..core.security import get_current_superuser
from ..models.database import User, Prediction, APIKey, UsageStats

router = APIRouter()


@router.get("/stats/platform")
async def get_platform_stats(
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """Get overall platform statistics (admin only)."""
    total_users = db.query(func.count(User.id)).scalar()
    active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar()
    total_predictions = db.query(func.count(Prediction.id)).scalar()
    total_api_keys = db.query(func.count(APIKey.id)).scalar()

    return {
        "users": {
            "total": total_users,
            "active": active_users,
            "inactive": total_users - active_users
        },
        "predictions": {
            "total": total_predictions
        },
        "api_keys": {
            "total": total_api_keys,
            "active": db.query(func.count(APIKey.id)).filter(APIKey.is_active == True).scalar()
        }
    }


@router.get("/users")
async def list_all_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """List all users (admin only)."""
    users = db.query(User).offset(skip).limit(limit).all()
    return users
