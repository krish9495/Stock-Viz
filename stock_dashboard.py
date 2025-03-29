import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import ta
from typing import Optional, Tuple, Dict, List

# Set page config
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define Indian stock symbols (you can add more)
INDIAN_STOCKS = {
    # Banking & Financial Services
    'HDFCBANK.NS': 'HDFC Bank',
    'ICICIBANK.NS': 'ICICI Bank',
    'SBIN.NS': 'State Bank of India',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'AXISBANK.NS': 'Axis Bank',
    'BAJFINANCE.NS': 'Bajaj Finance',
    'HDFC.NS': 'Housing Development Finance Corp',
    
    # Information Technology
    'TCS.NS': 'Tata Consultancy Services',
    'INFY.NS': 'Infosys',
    'WIPRO.NS': 'Wipro',
    'TECHM.NS': 'Tech Mahindra',
    'HCLTECH.NS': 'HCL Technologies',
    
    # Energy & Oil
    'RELIANCE.NS': 'Reliance Industries',
    'ONGC.NS': 'Oil & Natural Gas Corporation',
    'GAIL.NS': 'GAIL India',
    'ADANIGREEN.NS': 'Adani Green Energy',
    'TATAPOWER.NS': 'Tata Power',
    
    # Automobile
    'TATAMOTORS.NS': 'Tata Motors',
    'MARUTI.NS': 'Maruti Suzuki',
    'BAJAJ-AUTO.NS': 'Bajaj Auto',
    'HEROMOTOCO.NS': 'Hero MotoCorp',
    'M&M.NS': 'Mahindra & Mahindra',
    
    # Consumer Goods
    'ITC.NS': 'ITC',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'BRITANNIA.NS': 'Britannia Industries',
    'MARICO.NS': 'Marico',
    'TATACONSUM.NS': 'Tata Consumer Products',
    
    # Telecommunications
    'BHARTIARTL.NS': 'Bharti Airtel',
    'VODAFONEIDEA.NS': 'Vodafone Idea',
    
    # Metals & Mining
    'TATASTEEL.NS': 'Tata Steel',
    'JSWSTEEL.NS': 'JSW Steel',
    'SAIL.NS': 'Steel Authority of India',
    'HINDALCO.NS': 'Hindalco Industries',
    
    # Pharmaceuticals
    'SUNPHARMA.NS': 'Sun Pharmaceutical',
    'DRREDDY.NS': 'Dr. Reddy\'s Laboratories',
    'CIPLA.NS': 'Cipla',
    'APOLLOHOSP.NS': 'Apollo Hospitals',
    
    # Infrastructure
    'LT.NS': 'Larsen & Toubro',
    'ADANIENT.NS': 'Adani Enterprises',
    'ULTRACEMCO.NS': 'UltraTech Cement',
    'ACC.NS': 'ACC Limited',
    
    # FMCG & Retail
    'DMART.NS': 'Avenue Supermarts',
    'TITAN.NS': 'Titan Company',
    'DABUR.NS': 'Dabur India',
    
    # Others
    'ASIANPAINT.NS': 'Asian Paints',
    'NESTLEIND.NS': 'Nestle India',
    'BAJAJFINSV.NS': 'Bajaj Finserv',
    'ADANIPORTS.NS': 'Adani Ports',
    'POWERGRID.NS': 'Power Grid Corporation'
}

# Cache the stock data to improve performance
@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance with error handling and caching.
    
    Args:
        ticker: Stock symbol
        period: Time period (e.g., '1d', '5d', '1mo')
        interval: Data interval (e.g., '1m', '5m', '1h')
    
    Returns:
        DataFrame with stock data or None if error occurs
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            st.error(f"No data found for {ticker}. Please check the ticker symbol and try again.")
            return None
            
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_usd_to_inr_rate() -> float:
    """
    Get the current USD to INR conversion rate.
    
    Returns:
        Current USD to INR rate
    """
    try:
        usd_inr = yf.Ticker("USDINR=X")
        rate = usd_inr.history(period="1d")['Close'].iloc[-1]
        return rate
    except:
        return 83.0  # Fallback rate if API fails

def is_indian_stock(ticker: str) -> bool:
    """
    Check if the stock is an Indian stock.
    
    Args:
        ticker: Stock symbol
    
    Returns:
        True if it's an Indian stock, False otherwise
    """
    return ticker.endswith('.NS') or ticker in INDIAN_STOCKS

def get_stock_currency(ticker: str) -> str:
    """
    Get the currency for a given stock.
    
    Args:
        ticker: Stock symbol
    
    Returns:
        Currency code ('USD' or 'INR')
    """
    return 'INR' if is_indian_stock(ticker) else 'USD'

def convert_to_inr(usd_value: float) -> float:
    """
    Convert USD value to INR.
    
    Args:
        usd_value: Value in USD
    
    Returns:
        Value in INR
    """
    rate = get_usd_to_inr_rate()
    return usd_value * rate

def format_currency(value: float, currency: str = 'USD') -> str:
    """
    Format currency value with appropriate symbol.
    
    Args:
        value: The value to format
        currency: Currency code ('USD' or 'INR')
    
    Returns:
        Formatted string with currency symbol
    """
    if currency == 'USD':
        return f"${value:,.2f}"
    else:  # INR
        return f"₹{value:,.2f}"

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process and format the stock data.
    
    Args:
        data: Raw stock data DataFrame
    
    Returns:
        Processed DataFrame with proper timezone and formatting
    """
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

def calculate_metrics(data: pd.DataFrame) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate key metrics from stock data.
    
    Args:
        data: Processed stock data DataFrame
    
    Returns:
        Tuple of (last_close, change, pct_change, high, low, volume)
    """
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data['High'].max()
    low = data['Low'].min()
    volume = data['Volume'].sum()
    return last_close, change, pct_change, high, low, volume

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the stock data.
    
    Args:
        data: Processed stock data DataFrame
    
    Returns:
        DataFrame with added technical indicators
    """
    # Trend indicators
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    data['MACD'] = ta.trend.macd_diff(data['Close'])
    
    # Momentum indicators
    data['RSI_14'] = ta.momentum.rsi(data['Close'], window=14)
    data['Stoch'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
    
    # Volume indicators
    data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
    
    return data

def create_chart(data: pd.DataFrame, chart_type: str, indicators: List[str], ticker: str, time_period: str, currency: str = 'USD') -> go.Figure:
    """
    Create an interactive chart with selected indicators.
    
    Args:
        data: Processed stock data with indicators
        chart_type: Type of chart ('Candlestick' or 'Line')
        indicators: List of selected technical indicators
        ticker: Stock symbol
        time_period: Selected time period
        currency: Currency to display ('USD' or 'INR')
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Convert prices to INR if needed (only for US stocks)
    if currency == 'INR' and not is_indian_stock(ticker):
        rate = get_usd_to_inr_rate()
        data['Open'] = data['Open'] * rate
        data['High'] = data['High'] * rate
        data['Low'] = data['Low'] * rate
        data['Close'] = data['Close'] * rate
        data['SMA_20'] = data['SMA_20'] * rate
        data['EMA_20'] = data['EMA_20'] * rate
    
    # Add main price chart
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=data['Datetime'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=data['Datetime'],
            y=data['Close'],
            name='Price',
            line=dict(color='blue')
        ))
    
    # Add selected indicators
    for indicator in indicators:
        if indicator == 'SMA 20':
            fig.add_trace(go.Scatter(
                x=data['Datetime'],
                y=data['SMA_20'],
                name='SMA 20',
                line=dict(color='orange')
            ))
        elif indicator == 'EMA 20':
            fig.add_trace(go.Scatter(
                x=data['Datetime'],
                y=data['EMA_20'],
                name='EMA 20',
                line=dict(color='green')
            ))
        elif indicator == 'RSI 14':
            fig.add_trace(go.Scatter(
                x=data['Datetime'],
                y=data['RSI_14'],
                name='RSI 14',
                yaxis='y2',
                line=dict(color='purple')
            ))
        elif indicator == 'MACD':
            fig.add_trace(go.Scatter(
                x=data['Datetime'],
                y=data['MACD'],
                name='MACD',
                yaxis='y3',
                line=dict(color='red')
            ))
    
    # Update layout
    currency_symbol = '₹' if currency == 'INR' else '$'
    fig.update_layout(
        title=f"{ticker} {time_period.upper()} Chart",
        xaxis_title='Time',
        yaxis_title=f'Price ({currency_symbol})',
        yaxis2=dict(
            title='RSI',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, 100]
        ),
        yaxis3=dict(
            title='MACD',
            overlaying='y',
            side='right',
            showgrid=False,
            position=0.95
        ),
        height=600,
        template='plotly_dark'
    )
    
    return fig

def main():
    # Title and description
    st.title('📈 Real-Time Stock Market Dashboard')
    st.markdown("""
    This dashboard provides real-time stock market data visualization with technical indicators.
    Use the sidebar to customize your view.
    """)
    
    # Sidebar configuration
    st.sidebar.header('Chart Parameters')
    
    # Stock selection
    stock_type = st.sidebar.selectbox('Stock Market', ['US Stocks', 'Indian Stocks'])
    
    if stock_type == 'Indian Stocks':
        ticker = st.sidebar.selectbox('Select Indian Stock', list(INDIAN_STOCKS.keys()))
        currency = 'INR'  # Force INR for Indian stocks
    else:
        ticker = st.sidebar.text_input('Enter US Stock Symbol', 'AAPL').upper()
        currency = st.sidebar.selectbox('Currency', ['USD', 'INR'])
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        'Time Period',
        ['1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max'],
        index=2
    )
    
    # Chart type selection
    chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
    
    # Technical indicators selection
    available_indicators = ['SMA 20', 'EMA 20', 'RSI 14', 'MACD']
    indicators = st.sidebar.multiselect(
        'Technical Indicators',
        available_indicators,
        default=['SMA 20', 'RSI 14']
    )
    
    # Interval mapping
    interval_mapping = {
        '1d': '1m',
        '5d': '5m',
        '1mo': '1h',
        '3mo': '1d',
        '6mo': '1d',
        '1y': '1wk',
        '5y': '1mo',
        'max': '1mo',
    }
    
    # Update button
    if st.sidebar.button('Update Chart'):
        with st.spinner('Fetching data...'):
            data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
            
            if data is not None:
                # Process data
                data = process_data(data)
                data = add_technical_indicators(data)
                
                # Calculate metrics
                last_close, change, pct_change, high, low, volume = calculate_metrics(data)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    label=f"{ticker} Last Price",
                    value=format_currency(last_close if currency == 'USD' or is_indian_stock(ticker) else convert_to_inr(last_close), currency),
                    delta=f"{format_currency(change if currency == 'USD' or is_indian_stock(ticker) else convert_to_inr(change), currency)} ({pct_change:.2f}%)"
                )
                col2.metric('High', format_currency(high if currency == 'USD' or is_indian_stock(ticker) else convert_to_inr(high), currency))
                col3.metric('Low', format_currency(low if currency == 'USD' or is_indian_stock(ticker) else convert_to_inr(low), currency))
                col4.metric('Volume', f"{volume:,}")
                
                # Create and display chart
                fig = create_chart(data, chart_type, indicators, ticker, time_period, currency)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display data tables
                tab1, tab2 = st.tabs(['Historical Data', 'Technical Indicators'])
                
                with tab1:
                    display_data = data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    if currency == 'INR' and not is_indian_stock(ticker):
                        rate = get_usd_to_inr_rate()
                        for col in ['Open', 'High', 'Low', 'Close']:
                            display_data[col] = display_data[col] * rate
                    st.dataframe(display_data, use_container_width=True)
                
                with tab2:
                    display_data = data[['Datetime', 'SMA_20', 'EMA_20', 'RSI_14', 'MACD']].copy()
                    if currency == 'INR' and not is_indian_stock(ticker):
                        rate = get_usd_to_inr_rate()
                        for col in ['SMA_20', 'EMA_20']:
                            display_data[col] = display_data[col] * rate
                    st.dataframe(display_data, use_container_width=True)
    
    # Real-time stock prices in sidebar
    st.sidebar.header('Real-Time Stock Prices')
    
    if stock_type == 'Indian Stocks':
        stock_symbols = list(INDIAN_STOCKS.keys())
    else:
        stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']
    
    for symbol in stock_symbols:
        if symbol != ticker:  # Skip the currently selected stock
            real_time_data = fetch_stock_data(symbol, '1d', '1m')
            if real_time_data is not None:
                real_time_data = process_data(real_time_data)
                last_price = real_time_data['Close'].iloc[-1]
                change = last_price - real_time_data['Open'].iloc[0]
                pct_change = (change / real_time_data['Open'].iloc[0]) * 100
                
                if currency == 'INR' and not is_indian_stock(symbol):
                    last_price = convert_to_inr(last_price)
                    change = convert_to_inr(change)
                
                st.sidebar.metric(
                    f"{symbol}",
                    format_currency(last_price, currency),
                    f"{format_currency(change, currency)} ({pct_change:.2f}%)"
                )
    
    # Footer
    st.sidebar.markdown('---')
    st.sidebar.markdown("""
    ### About
    This dashboard uses the Yahoo Finance API to provide real-time stock market data.
    Data is cached for 5 minutes to improve performance.
    """)

if __name__ == "__main__":
    main() 