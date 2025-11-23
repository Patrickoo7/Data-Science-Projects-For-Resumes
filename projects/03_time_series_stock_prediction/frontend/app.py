"""
StockPredict Streamlit Dashboard.
Interactive time series forecasting visualization.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# Configuration
API_URL = "http://localhost:8000/api/v1"
API_KEY = st.secrets.get("api_key", "") if hasattr(st, 'secrets') else ""

# Page config
st.set_page_config(
    page_title="StockPredict - AI Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def make_api_request(endpoint, method="GET", data=None):
    """Make API request with error handling."""
    headers = {"X-API-Key": API_KEY}

    try:
        if method == "GET":
            response = requests.get(f"{API_URL}{endpoint}", headers=headers)
        else:
            response = requests.post(f"{API_URL}{endpoint}", json=data, headers=headers)

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def plot_forecast(historical_data, predictions, confidence_intervals, ticker):
    """Create interactive forecast plot."""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Historical prices
    if historical_data:
        df = pd.DataFrame(historical_data)
        df['Date'] = pd.to_datetime(df['Date'])

        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Close'],
                name='Historical Price',
                line=dict(color='blue', width=2)
            ),
            secondary_y=False
        )

    # Predictions
    if predictions:
        pred_dates = pd.date_range(
            start=datetime.now(),
            periods=len(predictions['predictions']),
            freq='D'
        )

        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=predictions['predictions'],
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ),
            secondary_y=False
        )

        # Confidence intervals
        if 'confidence_intervals' in predictions:
            fig.add_trace(
                go.Scatter(
                    x=pred_dates,
                    y=predictions['confidence_intervals']['upper'],
                    fill=None,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ),
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(
                    x=pred_dates,
                    y=predictions['confidence_intervals']['lower'],
                    fill='tonexty',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(255,0,0,0.2)',
                    name='95% Confidence'
                ),
                secondary_y=False
            )

    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=500
    )

    return fig


def main():
    """Main Streamlit app."""

    # Header
    st.markdown('<div class="main-header">📈 StockPredict AI</div>', unsafe_allow_html=True)
    st.markdown("### Production-Ready Stock Price Forecasting Platform")

    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")

        # API Key input
        api_key_input = st.text_input(
            "API Key",
            value=API_KEY,
            type="password",
            help="Enter your StockPredict API key"
        )

        if api_key_input:
            global API_KEY
            API_KEY = api_key_input

        st.divider()

        # Stock selection
        st.header("📊 Stock Selection")
        supported_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"]
        ticker = st.selectbox("Select Ticker", supported_tickers)

        forecast_days = st.slider(
            "Forecast Days",
            min_value=7,
            max_value=90,
            value=30,
            step=7
        )

        predict_button = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

        st.divider()
        st.header("ℹ️ About")
        st.info(
            "StockPredict uses LSTM deep learning models to forecast stock prices. "
            "Predictions include confidence intervals based on historical volatility."
        )

    # Main content
    tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Analytics", "🔬 Backtest"])

    with tab1:
        if predict_button and API_KEY:
            with st.spinner("Generating forecast..."):
                # Make prediction
                result = make_api_request(
                    "/forecast/predict",
                    method="POST",
                    data={
                        "ticker": ticker,
                        "forecast_days": forecast_days,
                        "use_cache": True
                    }
                )

                if result:
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Current Price",
                            f"${result['last_price']:.2f}"
                        )

                    with col2:
                        st.metric(
                            f"{forecast_days}-Day Forecast",
                            f"${result['final_prediction']:.2f}",
                            f"{result['percent_change']:.2f}%"
                        )

                    with col3:
                        st.metric(
                            "Model Type",
                            result['model_type'].upper()
                        )

                    with col4:
                        st.metric(
                            "Processing Time",
                            f"{result['processing_time_ms']:.0f}ms"
                        )

                    # Fetch historical data
                    historical = make_api_request(f"/forecast/historical/{ticker}")

                    # Plot forecast
                    fig = plot_forecast(
                        historical['data'] if historical else None,
                        result,
                        result.get('confidence_intervals'),
                        ticker
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Prediction details
                    with st.expander("📋 Detailed Predictions"):
                        pred_df = pd.DataFrame({
                            'Date': result['prediction_dates'],
                            'Predicted Price': [f"${p:.2f}" for p in result['predictions']],
                            'Upper Bound': [f"${u:.2f}" for u in result['confidence_intervals']['upper']],
                            'Lower Bound': [f"${l:.2f}" for l in result['confidence_intervals']['lower']]
                        })
                        st.dataframe(pred_df, use_container_width=True)

        elif not API_KEY:
            st.warning("⚠️ Please enter your API key in the sidebar to generate forecasts.")
        else:
            st.info("👈 Select a stock and click 'Generate Forecast' to begin.")

    with tab2:
        st.header("📊 Usage Analytics")

        if API_KEY:
            # Fetch analytics
            analytics = make_api_request("/analytics/usage/summary?days=30")

            if analytics:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Total Predictions (30 days)",
                        analytics.get('total_predictions', 0)
                    )

                with col2:
                    st.metric(
                        "Avg Processing Time",
                        f"{analytics.get('average_processing_time_ms', 0):.0f}ms"
                    )

                with col3:
                    st.metric(
                        "Most Predicted",
                        analytics.get('most_predicted_ticker', 'N/A')
                    )

                # Ticker breakdown
                if 'ticker_breakdown' in analytics:
                    st.subheader("Ticker Breakdown")
                    ticker_df = pd.DataFrame(
                        analytics['ticker_breakdown'].items(),
                        columns=['Ticker', 'Predictions']
                    )
                    st.bar_chart(ticker_df.set_index('Ticker'))

    with tab3:
        st.header("🔬 Backtest Strategy")

        col1, col2 = st.columns(2)

        with col1:
            backtest_ticker = st.selectbox("Ticker", supported_tickers, key="backtest")
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))

        with col2:
            initial_capital = st.number_input("Initial Capital ($)", value=10000.0, step=1000.0)
            end_date = st.date_input("End Date", value=datetime.now())

        if st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                result = make_api_request(
                    "/backtest/run",
                    method="POST",
                    data={
                        "ticker": backtest_ticker,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "initial_capital": initial_capital,
                        "strategy": "long_only"
                    }
                )

                if result:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Final Value", f"${result['final_value']:.2f}")

                    with col2:
                        st.metric("Total Return", f"{result['total_return']*100:.2f}%")

                    with col3:
                        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")

                    with col4:
                        st.metric("Max Drawdown", f"{result['max_drawdown']*100:.2f}%")

                    st.success("✅ Backtest completed successfully!")


if __name__ == "__main__":
    main()
