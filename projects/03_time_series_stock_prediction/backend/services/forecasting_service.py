"""
Stock price forecasting service with LSTM/GRU/Transformer models.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple
import json

from ..core.config import settings
from ..src.model import create_model


class StockForecaster:
    """Stock price forecasting service."""

    def __init__(self):
        self.device = torch.device(
            settings.DEVICE if torch.cuda.is_available() else 'cpu'
        )
        self.model = None
        self.scaler = None
        self.model_version = settings.APP_VERSION
        self.sequence_length = settings.SEQUENCE_LENGTH

    def load_model(self, model_type: str = None):
        """Load trained model."""
        if self.model is not None:
            return

        model_type = model_type or settings.MODEL_TYPE
        print(f"Loading {model_type} model...")
        print(f"Device: {self.device}")

        try:
            # Load model
            self.model = create_model(
                model_type=model_type,
                input_size=10,  # Adjusted based on features
                output_size=1
            )

            # Try to load checkpoint
            checkpoint = torch.load(settings.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded fine-tuned model")
        except Exception as e:
            print(f"Using initialized model: {e}")

        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    def fetch_stock_data(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch historical stock data from yfinance."""
        if start_date is None:
            start_date = settings.DATA_START_DATE

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Fetching {ticker} data from {start_date} to {end_date}")

        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')

        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, object]:
        """Prepare data for model input."""
        from sklearn.preprocessing import MinMaxScaler

        # Select features
        feature_columns = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI',
                          'MACD', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'EMA_12']

        data = df[feature_columns].values

        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        self.scaler = scaler

        return scaled_data, scaler

    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM input."""
        sequences = []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])

        return np.array(sequences)

    async def predict(
        self,
        ticker: str,
        forecast_days: int = 30,
        use_cache: bool = True
    ) -> Dict:
        """
        Predict stock prices.

        Args:
            ticker: Stock ticker symbol
            forecast_days: Number of days to forecast
            use_cache: Whether to use cached data

        Returns:
            Dict with predictions and metrics
        """
        if self.model is None:
            self.load_model()

        start_time = time.time()

        # Fetch and prepare data
        df = self.fetch_stock_data(ticker)
        df = self.calculate_technical_indicators(df)

        # Prepare data
        scaled_data, scaler = self.prepare_data(df)

        # Create sequences
        sequences = self.create_sequences(scaled_data)

        # Get last sequence for prediction
        last_sequence = sequences[-1:]
        last_sequence_tensor = torch.FloatTensor(last_sequence).to(self.device)

        # Generate predictions
        predictions = []
        current_sequence = last_sequence_tensor

        with torch.no_grad():
            for _ in range(forecast_days):
                # Predict next value
                pred = self.model(current_sequence)

                # Store prediction
                predictions.append(pred.cpu().numpy()[0][0])

                # Update sequence for next prediction
                new_row = current_sequence[0, -1, :].clone()
                new_row[0] = pred[0][0]  # Update close price

                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    new_row.unsqueeze(0).unsqueeze(0)
                ], dim=1)

        # Inverse transform predictions (only for close price column)
        predictions_array = np.array(predictions).reshape(-1, 1)

        # Create a dummy array for inverse transform
        dummy = np.zeros((len(predictions), scaled_data.shape[1]))
        dummy[:, 0] = predictions_array[:, 0]
        predictions_original = scaler.inverse_transform(dummy)[:, 0]

        # Calculate confidence intervals (simple method)
        historical_volatility = df['Close'].pct_change().std()
        confidence_intervals = {
            'upper': [float(p * (1 + 2 * historical_volatility)) for p in predictions_original],
            'lower': [float(p * (1 - 2 * historical_volatility)) for p in predictions_original]
        }

        # Get last actual price
        last_price = float(df['Close'].iloc[-1])

        # Calculate percentage change
        final_prediction = float(predictions_original[-1])
        percent_change = ((final_prediction - last_price) / last_price) * 100

        processing_time = (time.time() - start_time) * 1000

        result = {
            'ticker': ticker,
            'last_price': last_price,
            'forecast_days': forecast_days,
            'predictions': [float(p) for p in predictions_original],
            'confidence_intervals': confidence_intervals,
            'final_prediction': final_prediction,
            'percent_change': round(percent_change, 2),
            'model_type': settings.MODEL_TYPE,
            'model_version': self.model_version,
            'processing_time_ms': round(processing_time, 2),
            'prediction_dates': [
                (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
                for i in range(forecast_days)
            ]
        }

        return result

    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """Calculate prediction metrics."""
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }

    def get_model_info(self) -> Dict:
        """Get model information."""
        if self.model is None:
            self.load_model()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            'model_type': settings.MODEL_TYPE,
            'model_version': self.model_version,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'sequence_length': self.sequence_length,
            'supported_tickers': settings.ALLOWED_TICKERS
        }


# Global forecaster instance
forecaster = StockForecaster()
