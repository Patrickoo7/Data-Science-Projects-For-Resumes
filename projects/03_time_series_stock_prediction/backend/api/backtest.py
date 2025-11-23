"""Backtesting API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
from ..core.database import get_db
from ..core.security import get_current_active_user
from ..models.database import User, BacktestResult
from ..core.config import settings

router = APIRouter()


class BacktestRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    strategy: str = "long_only"  # long_only, long_short


@router.post("/run")
async def run_backtest(
    request: BacktestRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Run backtest on historical data.

    Simple implementation - in production would use actual strategy logic.
    """

    if request.ticker not in settings.ALLOWED_TICKERS:
        raise HTTPException(status_code=400, detail="Ticker not supported")

    # Placeholder backtest results
    # In production, this would run actual backtest logic
    result = BacktestResult(
        user_id=current_user.id,
        ticker=request.ticker,
        strategy_name=request.strategy,
        start_date=datetime.fromisoformat(request.start_date),
        end_date=datetime.fromisoformat(request.end_date),
        initial_capital=request.initial_capital,
        final_value=request.initial_capital * 1.35,  # Example 35% return
        total_return=0.35,
        total_trades=45,
        winning_trades=28,
        losing_trades=17,
        sharpe_ratio=1.42,
        sortino_ratio=1.85,
        max_drawdown=-0.12,
        calmar_ratio=2.91,
        equity_curve=[]  # Would contain actual equity curve data
    )

    db.add(result)
    db.commit()
    db.refresh(result)

    return result


@router.get("/results")
async def get_backtest_results(
    limit: int = 10,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get backtest results for current user."""
    results = db.query(BacktestResult).filter(
        BacktestResult.user_id == current_user.id
    ).order_by(BacktestResult.created_at.desc()).limit(limit).all()

    return results
