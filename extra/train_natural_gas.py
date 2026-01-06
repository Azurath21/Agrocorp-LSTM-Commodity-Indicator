"""
Natural Gas LSTM Model - Maximum Data Training
Focused on gas-related indicators and commodities
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, f1_score, precision_score, 
    recall_score, accuracy_score, confusion_matrix
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib

# ============================================================
# NATURAL GAS FOCUSED SYMBOLS
# ============================================================

# Target
TARGET = 'NG=F'  # Natural Gas Futures

# Gas-related commodities
GAS_RELATED = {
    'NG=F': 'Natural Gas Futures',
    'HO=F': 'Heating Oil',  # Competes with gas for heating
    'CL=F': 'Crude Oil',  # Energy correlation
    'BZ=F': 'Brent Crude',
    'RB=F': 'Gasoline',  # Petroleum products
    'UNG': 'Natural Gas ETF',
    'BOIL': 'Natural Gas 2x Bull ETF',
}

# Gas utility/producer stocks
GAS_STOCKS = {
    'XOM': 'Exxon Mobil',
    'CVX': 'Chevron',
    'COP': 'ConocoPhillips',
    'EOG': 'EOG Resources',
    'PXD': 'Pioneer Natural Resources',
    'DVN': 'Devon Energy',
    'SWN': 'Southwestern Energy',  # Major gas producer
    'EQT': 'EQT Corporation',  # Largest US gas producer
    'AR': 'Antero Resources',
    'RRC': 'Range Resources',
}

# Macro indicators
MACRO = {
    '^TNX': '10Y Treasury',
    '^VIX': 'Volatility Index',
    'DX-Y.NYB': 'US Dollar Index',
    'EURUSD=X': 'EUR/USD',
}

# Weather/seasonal (using utility ETFs as proxy)
WEATHER_PROXIES = {
    'XLU': 'Utilities Sector ETF',
    'XLE': 'Energy Sector ETF',
}


def fetch_max_data(symbol, min_years=10):
    """Fetch maximum available historical data"""
    try:
        ticker = yf.Ticker(symbol)
        # Try to get max data
        df = ticker.history(period='max')
        
        if df.empty:
            return None
        
        # Normalize index
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep='first')]
        
        # Rename columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = [f'{symbol}_{col}' for col in df.columns]
        
        return df
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return None


def calculate_technical_indicators(df, close_col, high_col, low_col, volume_col, prefix=''):
    """Calculate comprehensive technical indicators"""
    close = df[close_col]
    high = df[high_col]
    low = df[low_col]
    volume = df[volume_col]
    
    indicators = pd.DataFrame(index=df.index)
    
    # Trend indicators
    indicators[f'{prefix}SMA_10'] = ta.trend.sma_indicator(close, window=10)
    indicators[f'{prefix}SMA_20'] = ta.trend.sma_indicator(close, window=20)
    indicators[f'{prefix}SMA_50'] = ta.trend.sma_indicator(close, window=50)
    indicators[f'{prefix}EMA_10'] = ta.trend.ema_indicator(close, window=10)
    indicators[f'{prefix}EMA_20'] = ta.trend.ema_indicator(close, window=20)
    
    # Price vs MAs
    indicators[f'{prefix}price_vs_sma20'] = (close - indicators[f'{prefix}SMA_20']) / indicators[f'{prefix}SMA_20'] * 100
    indicators[f'{prefix}price_vs_sma50'] = (close - indicators[f'{prefix}SMA_50']) / indicators[f'{prefix}SMA_50'] * 100
    
    # MACD
    macd = ta.trend.MACD(close)
    indicators[f'{prefix}MACD'] = macd.macd()
    indicators[f'{prefix}MACD_signal'] = macd.macd_signal()
    indicators[f'{prefix}MACD_hist'] = macd.macd_diff()
    
    # RSI
    indicators[f'{prefix}RSI'] = ta.momentum.rsi(close, window=14)
    indicators[f'{prefix}RSI_7'] = ta.momentum.rsi(close, window=7)
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    indicators[f'{prefix}STOCH_k'] = stoch.stoch()
    indicators[f'{prefix}STOCH_d'] = stoch.stoch_signal()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close)
    indicators[f'{prefix}BB_high'] = bb.bollinger_hband()
    indicators[f'{prefix}BB_low'] = bb.bollinger_lband()
    indicators[f'{prefix}BB_pct'] = bb.bollinger_pband()
    
    # ATR
    indicators[f'{prefix}ATR'] = ta.volatility.average_true_range(high, low, close)
    
    # ADX
    adx = ta.trend.ADXIndicator(high, low, close)
    indicators[f'{prefix}ADX'] = adx.adx()
    
    # CCI
    indicators[f'{prefix}CCI'] = ta.trend.cci(high, low, close)
    
    # Williams %R
    indicators[f'{prefix}Williams_R'] = ta.momentum.williams_r(high, low, close)
    
    # ROC
    indicators[f'{prefix}ROC'] = ta.momentum.roc(close, window=10)
    
    # OBV
    indicators[f'{prefix}OBV'] = ta.volume.on_balance_volume(close, volume)
    
    # MFI
    indicators[f'{prefix}MFI'] = ta.volume.money_flow_index(high, low, close, volume)
    
    return indicators


def fetch_all_gas_data():
    """Fetch all natural gas related data with maximum history"""
    print("="*60)
    print("FETCHING NATURAL GAS DATA (MAXIMUM HISTORY)")
    print("="*60)
    
    # Fetch target commodity
    print(f"\n[Target: {TARGET}]")
    target_df = fetch_max_data(TARGET)
    if target_df is None:
        raise ValueError("Could not fetch Natural Gas data")
    
    print(f"  Data points: {len(target_df)}")
    print(f"  Date range: {target_df.index[0].strftime('%Y-%m-%d')} to {target_df.index[-1].strftime('%Y-%m-%d')}")
    
    combined_df = target_df.copy()
    
    # Calculate technical indicators for target
    print("\n[Calculating Technical Indicators]")
    tech_indicators = calculate_technical_indicators(
        target_df,
        f'{TARGET}_Close', f'{TARGET}_High', f'{TARGET}_Low', f'{TARGET}_Volume',
        prefix='NG_'
    )
    combined_df = pd.concat([combined_df, tech_indicators], axis=1)
    print(f"  Added {len(tech_indicators.columns)} technical indicators")
    
    # Fetch gas-related commodities
    print("\n[Fetching Gas-Related Commodities]")
    for symbol, name in GAS_RELATED.items():
        if symbol == TARGET:
            continue
        df = fetch_max_data(symbol)
        if df is not None:
            # Only use Close price
            col_name = f'{symbol}_Close'
            if col_name in df.columns:
                temp = df[[col_name]].reindex(combined_df.index, method='ffill')
                combined_df[col_name] = temp[col_name]
                print(f"  Added {name}: {df.notna().sum().iloc[0]} points")
    
    # Fetch gas stocks
    print("\n[Fetching Gas Producer Stocks]")
    for symbol, name in GAS_STOCKS.items():
        df = fetch_max_data(symbol)
        if df is not None:
            col_name = f'{symbol}_Close'
            if col_name in df.columns:
                temp = df[[col_name]].reindex(combined_df.index, method='ffill')
                combined_df[col_name] = temp[col_name]
                print(f"  Added {name}: {df.notna().sum().iloc[0]} points")
    
    # Fetch macro indicators
    print("\n[Fetching Macro Indicators]")
    for symbol, name in MACRO.items():
        df = fetch_max_data(symbol)
        if df is not None:
            col_name = f'{symbol}_Close'
            if col_name in df.columns:
                temp = df[[col_name]].reindex(combined_df.index, method='ffill')
                combined_df[col_name] = temp[col_name]
                print(f"  Added {name}: {df.notna().sum().iloc[0]} points")
    
    # Fetch weather proxies
    print("\n[Fetching Sector ETFs (Weather/Demand Proxies)]")
    for symbol, name in WEATHER_PROXIES.items():
        df = fetch_max_data(symbol)
        if df is not None:
            col_name = f'{symbol}_Close'
            if col_name in df.columns:
                temp = df[[col_name]].reindex(combined_df.index, method='ffill')
                combined_df[col_name] = temp[col_name]
                print(f"  Added {name}: {df.notna().sum().iloc[0]} points")
    
    # Add seasonal features
    print("\n[Adding Seasonal Features]")
    combined_df['month'] = combined_df.index.month
    combined_df['month_sin'] = np.sin(2 * np.pi * combined_df['month'] / 12)
    combined_df['month_cos'] = np.cos(2 * np.pi * combined_df['month'] / 12)
    combined_df['is_winter'] = combined_df['month'].isin([11, 12, 1, 2, 3]).astype(int)
    combined_df['is_summer'] = combined_df['month'].isin([6, 7, 8]).astype(int)
    combined_df = combined_df.drop('month', axis=1)
    print("  Added month_sin, month_cos, is_winter, is_summer")
    
    # Clean data
    combined_df = combined_df.ffill().bfill()
    combined_df = combined_df.dropna()
    
    print(f"\n{'='*60}")
    print(f"FINAL DATASET: {len(combined_df)} data points, {len(combined_df.columns)} features")
    print(f"Date range: {combined_df.index[0].strftime('%Y-%m-%d')} to {combined_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"{'='*60}")
    
    return combined_df


def prepare_lstm_data(df, sequence_length=60, prediction_days=5):
    """Prepare data for LSTM training"""
    feature_names = [col for col in df.columns]
    
    # Create target: direction and magnitude
    close_col = f'{TARGET}_Close'
    future_returns = df[close_col].pct_change(prediction_days).shift(-prediction_days) * 100
    
    # Direction: 1 if price goes up, 0 if down
    y_direction = (future_returns > 0).astype(int)
    y_pct_change = future_returns
    
    # Scale features
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Create sequences
    X, y_dir, y_pct = [], [], []
    
    for i in range(sequence_length, len(scaled_data) - prediction_days):
        X.append(scaled_data[i-sequence_length:i])
        y_dir.append(y_direction.iloc[i])
        y_pct.append(y_pct_change.iloc[i])
    
    X = np.array(X)
    y_dir = np.array(y_dir)
    y_pct = np.array(y_pct)
    
    # Remove any NaN
    valid_idx = ~(np.isnan(y_dir) | np.isnan(y_pct))
    X = X[valid_idx]
    y_dir = y_dir[valid_idx]
    y_pct = y_pct[valid_idx]
    
    print(f"\nLSTM Data Shape:")
    print(f"  X: {X.shape} (samples, sequence_length, features)")
    print(f"  y_direction: {y_dir.shape}")
    print(f"  y_pct_change: {y_pct.shape}")
    
    return X, y_dir, y_pct, feature_names, scaler


def create_lstm_model(seq_len, n_features, dropout=0.3, l2_reg=0.001):
    """Create LSTM model with anti-overfitting"""
    inputs = Input(shape=(seq_len, n_features))
    
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(l2_reg)))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    
    x = LSTM(32, return_sequences=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout * 0.5)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    
    direction = Dense(1, activation='sigmoid', name='direction')(x)
    magnitude = Dense(1, activation='linear', name='magnitude')(x)
    
    model = Model(inputs=inputs, outputs=[direction, magnitude])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'direction': 'binary_crossentropy', 'magnitude': 'huber'},
        loss_weights={'direction': 1.0, 'magnitude': 0.3},
        metrics={'direction': 'accuracy', 'magnitude': 'mae'}
    )
    
    return model


def evaluate_model(model, X_test, y_dir_test, y_pct_test):
    """Calculate comprehensive metrics"""
    direction_probs, magnitude_preds = model.predict(X_test, verbose=0)
    direction_probs = direction_probs.flatten()
    magnitude_preds = magnitude_preds.flatten()
    direction_preds = (direction_probs > 0.5).astype(int)
    y_true = y_dir_test.flatten()
    
    metrics = {
        'accuracy': accuracy_score(y_true, direction_preds),
        'precision': precision_score(y_true, direction_preds, zero_division=0),
        'recall': recall_score(y_true, direction_preds, zero_division=0),
        'f1_score': f1_score(y_true, direction_preds, zero_division=0),
        'auc_roc': roc_auc_score(y_true, direction_probs),
        'brier': brier_score_loss(y_true, direction_probs),
        'mae': np.mean(np.abs(y_pct_test - magnitude_preds)),
        'rmse': np.sqrt(np.mean((y_pct_test - magnitude_preds) ** 2))
    }
    
    tn, fp, fn, tp = confusion_matrix(y_true, direction_preds).ravel()
    metrics['tp'] = tp
    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    
    return metrics


def main():
    print("\n" + "="*60)
    print("NATURAL GAS LSTM MODEL - MAXIMUM DATA TRAINING")
    print("="*60)
    
    # Fetch data
    df = fetch_all_gas_data()
    
    # Prepare LSTM data
    sequence_length = 60
    prediction_days = 5
    X, y_dir, y_pct, feature_names, scaler = prepare_lstm_data(df, sequence_length, prediction_days)
    
    # Split: 70% train, 15% val, 15% test
    X_trainval, X_test, y_dir_trainval, y_dir_test, y_pct_trainval, y_pct_test = train_test_split(
        X, y_dir, y_pct, test_size=0.15, shuffle=False
    )
    X_train, X_val, y_dir_train, y_dir_val, y_pct_train, y_pct_val = train_test_split(
        X_trainval, y_dir_trainval, y_pct_trainval, test_size=0.18, shuffle=False
    )
    
    print(f"\nData Split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    
    # Create and train model
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    
    model = create_lstm_model(X.shape[1], X.shape[2], dropout=0.35, l2_reg=0.001)
    print(f"Model parameters: {model.count_params():,}")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)
    ]
    
    history = model.fit(
        X_train,
        {'direction': y_dir_train, 'magnitude': y_pct_train},
        validation_data=(X_val, {'direction': y_dir_val, 'magnitude': y_pct_val}),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    epochs_trained = len(history.history['loss'])
    print(f"\nEpochs trained: {epochs_trained}")
    
    # Evaluate
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    
    metrics = evaluate_model(model, X_test, y_dir_test, y_pct_test)
    
    print("\n--- Direction Prediction ---")
    print(f"Accuracy:     {metrics['accuracy']*100:.2f}%")
    print(f"Precision:    {metrics['precision']*100:.2f}%")
    print(f"Recall:       {metrics['recall']*100:.2f}%")
    print(f"F1 Score:     {metrics['f1_score']*100:.2f}%")
    print(f"AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"Brier Score:  {metrics['brier']:.4f}")
    
    print("\n--- Magnitude Prediction ---")
    print(f"MAE:          {metrics['mae']:.3f}%")
    print(f"RMSE:         {metrics['rmse']:.3f}%")
    
    print("\n--- Confusion Matrix ---")
    print(f"True Positives:  {metrics['tp']}")
    print(f"True Negatives:  {metrics['tn']}")
    print(f"False Positives: {metrics['fp']}")
    print(f"False Negatives: {metrics['fn']}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/natural_gas_lstm.keras')
    joblib.dump(scaler, 'models/natural_gas_scaler.joblib')
    joblib.dump({
        'commodity': TARGET,
        'sequence_length': sequence_length,
        'prediction_days': prediction_days,
        'feature_names': feature_names,
        'n_features': X.shape[2],
        'metrics': metrics,
        'epochs_trained': epochs_trained,
        'data_points': len(df)
    }, 'models/natural_gas_config.joblib')
    
    print(f"\nModel saved to models/natural_gas_lstm.keras")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
