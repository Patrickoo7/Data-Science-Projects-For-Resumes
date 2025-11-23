"""
Stock forecasting API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

from ..core.database import get_db
from ..core.security import get_current_active_user
from ..models.database import User, Prediction, StockData
from ..services.forecasting_service import forecaster
from ..core.config import settings

router = APIRouter()


# Pydantic models
class ForecastRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    forecast_days: int = Field(30, ge=1, le=90, description="Number of days to forecast")
    use_cache: bool = Field(True, description="Use cached data if available")


class ForecastResponse(BaseModel):
    ticker: str
    last_price: float
    forecast_days: int
    predictions: List[float]
    confidence_intervals: Dict[str, List[float]]
    final_prediction: float
    percent_change: float
    model_type: str
    model_version: str
    processing_time_ms: float
    prediction_dates: List[str]


class HistoricalDataRequest(BaseModel):
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# Background task to save prediction
async def save_prediction_task(
    user_id: int,
    result: dict,
    db: Session
):
    """Save prediction to database."""
    try:
        prediction = Prediction(
            user_id=user_id,
            ticker=result['ticker'],
            forecast_days=result['forecast_days'],
            forecast_values=result['predictions'],
            confidence_intervals=result['confidence_intervals'],
            model_type=result['model_type'],
            model_version=result['model_version'],
            processing_time_ms=result['processing_time_ms']
        )
        db.add(prediction)
        db.commit()
    except Exception as e:
        print(f"Error saving prediction: {e}")
        db.rollback()


@router.post("/predict", response_model=ForecastResponse)
async def predict_stock_price(
    request: ForecastRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Predict future stock prices.

    - **ticker**: Stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
    - **forecast_days**: Number of days to forecast (1-90)
    - **use_cache**: Whether to use cached historical data

    Returns predictions with confidence intervals and metrics.
    """

    # Validate ticker
    if request.ticker.upper() not in settings.ALLOWED_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Ticker {request.ticker} not supported. Allowed: {settings.ALLOWED_TICKERS}"
        )

    try:
        # Perform prediction
        result = await forecaster.predict(
            ticker=request.ticker.upper(),
            forecast_days=request.forecast_days,
            use_cache=request.use_cache
        )

        # Save prediction in background
        background_tasks.add_task(save_prediction_task, current_user.id, result, db)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/historical/{ticker}")
async def get_historical_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get historical stock data with technical indicators.

    - **ticker**: Stock ticker symbol
    - **start_date**: Start date (YYYY-MM-DD)
    - **end_date**: End date (YYYY-MM-DD)

    Returns OHLCV data with technical indicators.
    """

    if ticker.upper() not in settings.ALLOWED_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Ticker {ticker} not supported"
        )

    try:
        df = forecaster.fetch_stock_data(ticker.upper(), start_date, end_date)
        df = forecaster.calculate_technical_indicators(df)

        # Convert to dict
        data = df.reset_index().to_dict(orient='records')

        return {
            "ticker": ticker.upper(),
            "data_points": len(data),
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "data": data[:100]  # Limit to 100 records for response size
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {str(e)}")


@router.get("/model/info")
async def get_model_info(current_user: User = Depends(get_current_active_user)):
    """
    Get information about the forecasting model.

    Returns model architecture, parameters, and configuration.
    """
    return forecaster.get_model_info()


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


@router.get("/tickers/supported")
async def get_supported_tickers(current_user: User = Depends(get_current_active_user)):
    """Get list of supported stock tickers."""
    return {
        "supported_tickers": settings.ALLOWED_TICKERS,
        "total_count": len(settings.ALLOWED_TICKERS)
    }
