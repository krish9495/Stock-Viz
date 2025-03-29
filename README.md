# Stock Market Visualizer Dashboard

A powerful and interactive stock market dashboard built with Streamlit that allows users to analyze and visualize stock data from both Indian and US markets.

## Features

- 📈 Real-time stock data visualization
- 🌍 Support for both Indian and US stock markets
- 📊 Multiple technical indicators:
  - Moving Averages (SMA, EMA)
  - MACD
  - RSI
  - Stochastic Oscillator
  - On-Balance Volume (OBV)
- 💰 Currency conversion (USD to INR)
- 📱 Responsive and interactive charts
- 🔄 Auto-refreshing data
- 📅 Multiple time period selections
- 📊 Candlestick charts with volume
- 📈 Price and volume analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock_market_visualizer.git
cd stock_market_visualizer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run stock_dashboard.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Select a stock from the dropdown menu and choose your preferred time period and interval

4. Explore the various charts and technical indicators

## Features in Detail

### Stock Selection
- Choose from a comprehensive list of Indian stocks across various sectors
- Support for US stocks (enter ticker symbol directly)

### Technical Analysis
- Price charts with candlestick patterns
- Volume analysis
- Multiple technical indicators for better decision making
- Customizable chart settings

### Data Visualization
- Interactive charts using Plotly
- Multiple chart types (candlestick, line, area)
- Volume analysis
- Price movement indicators

## Dependencies

- streamlit >= 1.31.0
- yfinance >= 0.2.36
- pandas >= 2.2.0
- numpy >= 1.26.0
- plotly >= 5.18.0
- pytz >= 2024.1
- ta >= 0.11.0

## Deployment

This application is deployed on Streamlit Cloud. You can access it at:
[Your Streamlit Cloud URL]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data provided by Yahoo Finance
- Technical indicators powered by the `ta` library
- Visualization powered by Plotly
- UI framework by Streamlit

## Disclaimer

This tool is for educational and research purposes only. The information provided should not be construed as financial advice. Always do your own research and consult with financial advisors before making investment decisions.
