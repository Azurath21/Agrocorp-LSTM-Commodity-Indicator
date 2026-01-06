import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import warnings
warnings.filterwarnings('ignore')

ENERGY_COMMODITIES = {
    'CL=F': 'Crude Oil',
    'BZ=F': 'Brent Crude',
    'NG=F': 'Natural Gas',
    'HO=F': 'Heating Oil',
    'RB=F': 'Gasoline'
}

METAL_COMMODITIES = {
    'GC=F': 'Gold',
    'SI=F': 'Silver',
    'HG=F': 'Copper',
    'PL=F': 'Platinum',
    'PA=F': 'Palladium'
}

AGRICULTURE_COMMODITIES = {
    'ZC=F': 'Corn',
    'ZS=F': 'Soybean',
    'ZW=F': 'Wheat',
    'KC=F': 'Coffee',
    'CC=F': 'Cocoa',
    'CT=F': 'Cotton',
    'SB=F': 'Sugar'
}

ALL_COMMODITIES = {**ENERGY_COMMODITIES, **METAL_COMMODITIES, **AGRICULTURE_COMMODITIES}

MACRO_INDICATORS = {
    '^TNX': '10Y Treasury Yield',
    '^TYX': '30Y Treasury Yield',
    '^FVX': '5Y Treasury Yield',
    '^VIX': 'Volatility Index',
    'DX-Y.NYB': 'US Dollar Index',
    'EURUSD=X': 'EUR/USD',
    'CNY=X': 'USD/CNY'
}

SHIPPING_INDICATORS = {
    'BDRY': 'Dry Bulk Shipping ETF',
    'SBLK': 'Star Bulk Carriers',
    'GNK': 'Genco Shipping'
}

COMMODITY_ETFS = {
    'DBC': 'Commodity Index',
    'DBA': 'Agriculture ETF',
    'USO': 'Oil ETF',
    'GLD': 'Gold ETF'
}

FEATURE_CATEGORIES = {
    'price_movement': {
        'name': 'Price Movement',
        'description': 'OHLCV data for the target commodity',
        'features': []  # Populated dynamically
    },
    'technical_indicators': {
        'name': 'Technical Indicators',
        'description': 'RSI, MACD, Stochastic, ADX, Bollinger Bands, etc.',
        'features': []
    },
    'macro_indicators': {
        'name': 'Macro Indicators',
        'description': 'Treasury yields, VIX, Dollar Index, Forex rates',
        'features': []
    },
    'shipping_indicators': {
        'name': 'Shipping & Trade',
        'description': 'Dry bulk shipping, carrier stocks (global trade proxy)',
        'features': []
    },
    'related_commodities': {
        'name': 'Related Commodities',
        'description': 'Correlated commodity prices',
        'features': []
    }
}


def fetch_historical_data(symbol, years=5):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            print(f"Warning: No data for {symbol}")
            return None
        
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = [f'{symbol}_{col}' for col in df.columns]
        
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def calculate_technical_indicators(df, price_col, high_col, low_col, volume_col, prefix=''):
    indicators = pd.DataFrame(index=df.index)
    
    close = df[price_col]
    high = df[high_col]
    low = df[low_col]
    volume = df[volume_col]
    
    for period in [5, 10, 20, 50, 100, 200]:
        if len(close) >= period:
            indicators[f'{prefix}SMA_{period}'] = ta.trend.sma_indicator(close, window=period)
            indicators[f'{prefix}EMA_{period}'] = ta.trend.ema_indicator(close, window=period)
    
    indicators[f'{prefix}RSI_14'] = ta.momentum.rsi(close, window=14)
    
    macd = ta.trend.MACD(close)
    indicators[f'{prefix}MACD'] = macd.macd()
    indicators[f'{prefix}MACD_signal'] = macd.macd_signal()
    indicators[f'{prefix}MACD_hist'] = macd.macd_diff()
    
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    indicators[f'{prefix}STOCH_k'] = stoch.stoch()
    indicators[f'{prefix}STOCH_d'] = stoch.stoch_signal()
    
    indicators[f'{prefix}Williams_R'] = ta.momentum.williams_r(high, low, close)
    
    indicators[f'{prefix}CCI'] = ta.trend.cci(high, low, close)
    
    indicators[f'{prefix}ATR'] = ta.volatility.average_true_range(high, low, close)
    
    adx = ta.trend.ADXIndicator(high, low, close)
    indicators[f'{prefix}ADX'] = adx.adx()
    indicators[f'{prefix}ADX_pos'] = adx.adx_pos()
    indicators[f'{prefix}ADX_neg'] = adx.adx_neg()
    
    bb = ta.volatility.BollingerBands(close)
    indicators[f'{prefix}BB_high'] = bb.bollinger_hband()
    indicators[f'{prefix}BB_low'] = bb.bollinger_lband()
    indicators[f'{prefix}BB_mid'] = bb.bollinger_mavg()
    indicators[f'{prefix}BB_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    
    indicators[f'{prefix}ROC'] = ta.momentum.roc(close, window=12)
    
    indicators[f'{prefix}UO'] = ta.momentum.ultimate_oscillator(high, low, close)
    
    if volume.sum() > 0:
        indicators[f'{prefix}OBV'] = ta.volume.on_balance_volume(close, volume)
    
    if len(close) >= 50:
        indicators[f'{prefix}price_vs_SMA50'] = (close - indicators[f'{prefix}SMA_50']) / indicators[f'{prefix}SMA_50'] * 100
    
    if len(close) >= 200:
        indicators[f'{prefix}price_vs_SMA200'] = (close - indicators[f'{prefix}SMA_200']) / indicators[f'{prefix}SMA_200'] * 100
    
    return indicators


def fetch_all_data(target_commodity='CL=F', years=5):
    print(f"Fetching data for {target_commodity} with {years} years of history...")
    
    feature_categories = {
        'price_movement': [],
        'technical_indicators': [],
        'macro_indicators': [],
        'shipping_indicators': [],
        'related_commodities': []
    }
    
    target_df = fetch_historical_data(target_commodity, years)
    if target_df is None:
        raise ValueError(f"Could not fetch data for target commodity: {target_commodity}")
    
    print(f"Target commodity: {len(target_df)} data points")
    
    for col in target_df.columns:
        feature_categories['price_movement'].append(col)
    
    target_indicators = calculate_technical_indicators(
        target_df,
        f'{target_commodity}_Close',
        f'{target_commodity}_High',
        f'{target_commodity}_Low',
        f'{target_commodity}_Volume',
        prefix='target_'
    )
    
    for col in target_indicators.columns:
        feature_categories['technical_indicators'].append(col)
    
    combined_df = pd.concat([target_df, target_indicators], axis=1)
    
    print("Fetching macro indicators...")
    for symbol, name in MACRO_INDICATORS.items():
        df = fetch_historical_data(symbol, years)
        if df is not None:
            col_name = f'{symbol}_Close'
            temp_df = df[[col_name]].copy()
            temp_df = temp_df.reindex(combined_df.index, method='ffill')
            combined_df[col_name] = temp_df[col_name]
            feature_categories['macro_indicators'].append(col_name)
            print(f"  Added {name}")
    
    print("Fetching shipping indicators...")
    for symbol, name in SHIPPING_INDICATORS.items():
        df = fetch_historical_data(symbol, years)
        if df is not None:
            col_name = f'{symbol}_Close'
            temp_df = df[[col_name]].copy()
            temp_df = temp_df.reindex(combined_df.index, method='ffill')
            combined_df[col_name] = temp_df[col_name]
            feature_categories['shipping_indicators'].append(col_name)
            print(f"  Added {name}")
    
    print("Fetching related commodities...")
    related = list(ALL_COMMODITIES.keys())[:5]
    for symbol in related:
        if symbol != target_commodity:
            df = fetch_historical_data(symbol, years)
            if df is not None:
                col_name = f'{symbol}_Close'
                temp_df = df[[col_name]].copy()
                temp_df = temp_df.reindex(combined_df.index, method='ffill')
                combined_df[col_name] = temp_df[col_name]
                feature_categories['related_commodities'].append(col_name)
                print(f"  Added {ALL_COMMODITIES.get(symbol, symbol)}")
    
    combined_df = combined_df.ffill().bfill()
    
    combined_df = combined_df.dropna()
    
    print(f"\n{'='*50}")
    print("FEATURE SUMMARY BY CATEGORY")
    print(f"{'='*50}")
    print(f"Price Movement:      {len(feature_categories['price_movement'])} features")
    print(f"Technical Indicators: {len(feature_categories['technical_indicators'])} features")
    print(f"Macro Indicators:    {len(feature_categories['macro_indicators'])} features")
    print(f"Shipping & Trade:    {len(feature_categories['shipping_indicators'])} features")
    print(f"Related Commodities: {len(feature_categories['related_commodities'])} features")
    print(f"{'='*50}")
    print(f"TOTAL: {len(combined_df.columns)} features, {len(combined_df)} data points")
    
    return combined_df, feature_categories


def create_target_variable(df, target_col, prediction_days=5):
    future_price = df[target_col].shift(-prediction_days)
    current_price = df[target_col]
    
    pct_change = (future_price - current_price) / current_price * 100
    
    direction = (pct_change > 0).astype(int)
    
    magnitude = pct_change.abs()
    
    return direction, pct_change, magnitude


def prepare_lstm_data(df, target_commodity='CL=F', sequence_length=60, prediction_days=5):
    target_col = f'{target_commodity}_Close'
    
    direction, pct_change, magnitude = create_target_variable(df, target_col, prediction_days)
    
    df = df.copy()
    df['target_direction'] = direction
    df['target_pct_change'] = pct_change
    df['target_magnitude'] = magnitude
    
    df = df.iloc[:-prediction_days]
    
    feature_cols = [col for col in df.columns if not col.startswith('target_')]
    feature_names = feature_cols
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    
    X, y_direction, y_pct_change = [], [], []
    
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i-sequence_length:i])
        y_direction.append(df['target_direction'].iloc[i])
        y_pct_change.append(df['target_pct_change'].iloc[i])
    
    X = np.array(X)
    y_direction = np.array(y_direction)
    y_pct_change = np.array(y_pct_change)
    
    print(f"\nLSTM Data Shape:")
    print(f"  X: {X.shape} (samples, sequence_length, features)")
    print(f"  y_direction: {y_direction.shape}")
    print(f"  y_pct_change: {y_pct_change.shape}")
    
    return X, y_direction, y_pct_change, feature_names, scaler


if __name__ == "__main__":
    df = fetch_all_data('CL=F', years=5)
    print("\nSample of fetched data:")
    print(df.head())
    print("\nColumns:", df.columns.tolist()[:20], "...")
