"""
Stock data fetching and preprocessing.
Technical indicators calculation and feature engineering.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse


def fetch_stock_data(ticker: str, start_date: str, end_date: str):
    """Fetch historical stock data."""
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df


def calculate_technical_indicators(df: pd.DataFrame):
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

    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

    # Price momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['ROC'] = df['Close'].pct_change(periods=10) * 100

    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    return df


def create_sequences(data, seq_length=60):
    """Create sequences for LSTM training."""
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Predict closing price

    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date')
    parser.add_argument('--end', type=str, default=None, help='End date')
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime('%Y-%m-%d')

    print(f"Fetching data for {args.ticker} from {args.start} to {args.end}")

    # Fetch data
    df = fetch_stock_data(args.ticker, args.start, args.end)

    # Calculate indicators
    df = calculate_technical_indicators(df)

    # Save
    output_path = f'data/{args.ticker}.csv'
    df.to_csv(output_path)
    print(f"Data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
