# Agrocorp LSTM Commodity Indicator

Deep learning-based commodity price prediction using LSTM neural networks with technical indicators, macroeconomic data, and shipping indicators.

## Features

- **Multi-input LSTM Model**: Uses 60 days of historical data to predict 5-day price movement
- **Technical Indicators**: RSI, MACD, Stochastic, Williams %R, CCI, ADX, Bollinger Bands, ROC, etc.
- **Macro Indicators**: Treasury yields, VIX, Dollar Index, Forex rates
- **Shipping Indicators**: BDRY (Dry Bulk proxy for Baltic Dry Index), shipping stocks
- **Dual Output**: Predicts both direction (up/down) and magnitude (% change)
- **Web Interface**: Simple Flask-based UI for predictions
- **Explainability**: Category-level breakdown of what's driving predictions

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Models

```bash
# Train all commodities (Crude Oil, Natural Gas, Gold)
python train.py --commodity all

# Train a specific commodity
python train.py --commodity CL=F  # Crude Oil
python train.py --commodity NG=F  # Natural Gas
python train.py --commodity GC=F  # Gold
```

### 2. Run the Web App

```bash
python app.py
```

Open http://localhost:5000 in your browser.

## Supported Commodities

| Symbol | Name |
|--------|------|
| CL=F | Crude Oil |
| NG=F | Natural Gas |
| GC=F | Gold |

## Model Architecture

```
Input (60 timesteps x N features)
    ↓
Bidirectional LSTM (128 units) + BatchNorm + Dropout(0.3)
    ↓
Bidirectional LSTM (64 units) + BatchNorm + Dropout(0.3)
    ↓
LSTM (32 units) + BatchNorm + Dropout(0.2)
    ↓
Dense (64) + BatchNorm + Dropout(0.2)
    ↓
    ├── Direction Branch → Dense(32) → Sigmoid (UP/DOWN)
    └── Magnitude Branch → Dense(32) → Linear (% change)
```

## Data Sources

All data is fetched from Yahoo Finance API:

- **Commodity Prices**: OHLCV data
- **Technical Indicators**: Calculated using `ta` library
- **Macro Indicators**: ^TNX, ^TYX, ^FVX, ^VIX, DX-Y.NYB, EURUSD=X, CNY=X
- **Shipping Indicators**: BDRY, SBLK, GNK (proxies for global trade activity)

## Output

The model provides:
- **Direction**: UP or DOWN
- **Confidence**: 0-100% probability
- **Predicted Change**: Expected % price change
- **Target Price**: Calculated from current price + predicted change
- **Strength**: STRONG, MODERATE, or WEAK signal

## Files

```
Agrocorp-LSTM-Commodity-Indicator/
├── app.py                # Flask web application
├── data_fetcher.py       # Data fetching and preprocessing
├── lstm_model.py         # LSTM model architecture
├── train.py              # Training script
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Web UI
├── models/               # Saved models (created after training)
│   ├── CL_F_model.keras  # Crude Oil model
│   ├── NG_F_model.keras  # Natural Gas model
│   ├── GC_F_model.keras  # Gold model
│   └── *_scaler.joblib   # Feature scalers
└── extra/                # Additional experimental scripts
```

## Notes

- Training uses ~1,260 data points (5 years of daily data)
- Each commodity has its own trained model
- Model training takes ~10-30 minutes per commodity
- GPU acceleration supported if TensorFlow GPU is installed
- Predictions are for educational purposes only - not financial advice
