from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import joblib
import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import ta
import shap
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)

model = None
scaler = None
config = None
background_data = None

FEATURE_CATEGORIES = {
    'technical_indicators': {
        'name': 'Technical Indicators',
        'icon': 'chart-line',
        'keywords': ['SMA', 'EMA', 'RSI', 'MACD', 'STOCH', 'Williams', 'CCI', 'ATR', 'ADX', 'BB_', 'ROC', 'Ultimate', 'price_vs']
    },
    'macro_indicators': {
        'name': 'Macro Indicators', 
        'icon': 'landmark',
        'keywords': ['^TNX', '^TYX', '^FVX', '^VIX', 'DX-Y', 'EURUSD', 'CNY=X']
    },
    'shipping_trade': {
        'name': 'Shipping & Trade',
        'icon': 'ship',
        'keywords': ['BDRY', 'SBLK', 'GNK']
    },
    'related_commodities': {
        'name': 'Related Commodities',
        'icon': 'cubes',
        'keywords': ['CL=F', 'BZ=F', 'NG=F', 'HO=F', 'RB=F', 'GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F']
    },
    'price_movement': {
        'name': 'Price Movement',
        'icon': 'dollar-sign',
        'keywords': ['_Open', '_High', '_Low', '_Close', '_Volume']
    }
}


def categorize_feature(feature_name, target_commodity):
    if feature_name.startswith(target_commodity):
        for kw in ['_Open', '_High', '_Low', '_Close', '_Volume']:
            if kw in feature_name:
                return 'price_movement'
    
    for kw in FEATURE_CATEGORIES['technical_indicators']['keywords']:
        if kw in feature_name:
            return 'technical_indicators'
    
    for kw in FEATURE_CATEGORIES['macro_indicators']['keywords']:
        if kw in feature_name:
            return 'macro_indicators'
    
    for kw in FEATURE_CATEGORIES['shipping_trade']['keywords']:
        if kw in feature_name:
            return 'shipping_trade'
    
    for kw in FEATURE_CATEGORIES['related_commodities']['keywords']:
        if kw in feature_name and not feature_name.startswith(target_commodity):
            return 'related_commodities'
    
    return 'technical_indicators'


def calculate_feature_importance(X_input, feature_names, target_commodity):
    global model
    
    import tensorflow as tf
    
    X_tensor = tf.convert_to_tensor(X_input, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model(X_tensor, training=False)
        direction_output = predictions[0]
    
    gradients = tape.gradient(direction_output, X_tensor)
    
    feature_importance = np.abs(gradients.numpy()).mean(axis=1)[0]
    
    total = feature_importance.sum()
    if total > 0:
        feature_importance = feature_importance / total * 100
    
    category_contributions = {}
    for cat_key in FEATURE_CATEGORIES:
        category_contributions[cat_key] = {
            'name': FEATURE_CATEGORIES[cat_key]['name'],
            'icon': FEATURE_CATEGORIES[cat_key]['icon'],
            'contribution': 0.0,
            'features': []
        }
    
    for i, feat_name in enumerate(feature_names):
        category = categorize_feature(feat_name, target_commodity)
        importance = float(feature_importance[i])
        category_contributions[category]['contribution'] += importance
        if importance > 0.5:  
            category_contributions[category]['features'].append({
                'name': feat_name,
                'importance': round(importance, 2)
            })
    
    for cat_key in category_contributions:
        category_contributions[cat_key]['features'] = sorted(
            category_contributions[cat_key]['features'],
            key=lambda x: x['importance'],
            reverse=True
        )[:5]
        category_contributions[cat_key]['contribution'] = round(
            category_contributions[cat_key]['contribution'], 1
        )
    
    return category_contributions


def calculate_directional_contribution(X_input, feature_names, target_commodity, direction_prob):
    global model
    import tensorflow as tf
    
    is_bullish = direction_prob > 0.5
    n_features = len(feature_names)
    
    category_analysis = {}
    for cat_key in FEATURE_CATEGORIES:
        category_analysis[cat_key] = {
            'name': FEATURE_CATEGORIES[cat_key]['name'],
            'icon': FEATURE_CATEGORIES[cat_key]['icon'],
            'contribution_pct': 0.0,
            'direction': 'neutral',
            'supports_prediction': True,
            'top_features': []
        }
    
    category_indices = {cat: [] for cat in FEATURE_CATEGORIES}
    for i, feat_name in enumerate(feature_names):
        category = categorize_feature(feat_name, target_commodity)
        category_indices[category].append(i)
    
    X_tensor = tf.convert_to_tensor(X_input, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model(X_tensor, training=False)
        direction_output = predictions[0][0, 0]
    
    grads = tape.gradient(direction_output, X_tensor)
    if grads is None:
        return category_analysis
    
    grads_np = grads.numpy()
    
    feature_importance = np.abs(grads_np[0]).mean(axis=0)
    
    input_values = X_input[0]
    feature_direction = (grads_np[0] * input_values).mean(axis=0)
    
    total_importance = feature_importance.sum()
    if total_importance > 0:
        feature_importance_pct = feature_importance / total_importance * 100
    else:
        feature_importance_pct = np.zeros(n_features)
    
    for cat_key, indices in category_indices.items():
        if not indices:
            continue
        
        cat_importance = 0.0
        cat_direction_score = 0.0
        feat_details = []
        
        for idx in indices:
            imp = float(feature_importance_pct[idx])
            dir_score = float(feature_direction[idx])
            cat_importance += imp
            cat_direction_score += dir_score * imp
            
            if imp > 0.5:
                feat_details.append({
                    'name': feature_names[idx].replace('target_', '').replace('_Close', ''),
                    'importance': round(imp, 1)
                })
        
        if cat_direction_score > 0.001:
            cat_direction = 'bullish'
        elif cat_direction_score < -0.001:
            cat_direction = 'bearish'
        else:
            cat_direction = 'neutral'
        
        supports = (cat_direction == 'bullish' and is_bullish) or \
                   (cat_direction == 'bearish' and not is_bullish) or \
                   cat_direction == 'neutral'
        
        category_analysis[cat_key]['contribution_pct'] = round(cat_importance, 1)
        category_analysis[cat_key]['direction'] = cat_direction
        category_analysis[cat_key]['supports_prediction'] = supports
        
        feat_details.sort(key=lambda x: x['importance'], reverse=True)
        category_analysis[cat_key]['top_features'] = feat_details[:3]
    
    return category_analysis


def load_resources_for_commodity(symbol):
    global model, scaler, config
    
    try:
        import tensorflow as tf
        safe_symbol = symbol.replace('=', '_')
        
        model_path = f'models/{safe_symbol}_model.keras'
        scaler_path = f'models/{safe_symbol}_scaler.joblib'
        config_path = f'models/{safe_symbol}_config.joblib'
        
        if not os.path.exists(model_path):
            print(f"No model found for {symbol}. Please train first.")
            return False
        
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        config = joblib.load(config_path)
        print(f"Loaded model for {symbol}")
        return True
    except Exception as e:
        print(f"Error loading resources for {symbol}: {e}")
        return False


def load_resources():
    return load_resources_for_commodity('CL=F')


def fetch_recent_data(symbol, days=100):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 2)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            return None
        
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def calculate_indicators(df, prefix='target_'):
    indicators = pd.DataFrame(index=df.index)
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
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


def prepare_prediction_data(commodity_symbol):
    if config is None:
        return None, "Model not loaded"
    
    sequence_length = config['sequence_length']
    feature_names = config['feature_names']
    
    target_df = fetch_recent_data(commodity_symbol, days=sequence_length + 100)
    if target_df is None:
        return None, f"Could not fetch data for {commodity_symbol}"
    
    combined_df = pd.DataFrame(index=target_df.index)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        combined_df[f'{commodity_symbol}_{col}'] = target_df[col]
    
    indicators = calculate_indicators(target_df, prefix='target_')
    combined_df = pd.concat([combined_df, indicators], axis=1)
    
    macro_symbols = ['^TNX', '^TYX', '^FVX', '^VIX', 'DX-Y.NYB', 'EURUSD=X', 'CNY=X']
    for symbol in macro_symbols:
        try:
            df = fetch_recent_data(symbol, days=sequence_length + 50)
            if df is not None:
                col_name = f'{symbol}_Close'
                temp_series = df['Close'].reindex(combined_df.index, method='ffill')
                combined_df[col_name] = temp_series
        except:
            pass
    
    shipping_symbols = ['BDRY', 'SBLK', 'GNK']
    for symbol in shipping_symbols:
        try:
            df = fetch_recent_data(symbol, days=sequence_length + 50)
            if df is not None:
                col_name = f'{symbol}_Close'
                temp_series = df['Close'].reindex(combined_df.index, method='ffill')
                combined_df[col_name] = temp_series
        except:
            pass
    
    related = ['CL=F', 'GC=F', 'NG=F', 'SI=F', 'HG=F']
    for symbol in related:
        if symbol != commodity_symbol:
            try:
                df = fetch_recent_data(symbol, days=sequence_length + 50)
                if df is not None:
                    col_name = f'{symbol}_Close'
                    temp_series = df['Close'].reindex(combined_df.index, method='ffill')
                    combined_df[col_name] = temp_series
            except:
                pass
    
    combined_df = combined_df.ffill().bfill()
    combined_df = combined_df.dropna()
    
    missing_features = set(feature_names) - set(combined_df.columns)
    for feat in missing_features:
        combined_df[feat] = 0
    
    combined_df = combined_df[feature_names]
    
    if len(combined_df) < sequence_length:
        return None, f"Not enough data points. Need {sequence_length}, got {len(combined_df)}"
    
    recent_data = combined_df.iloc[-sequence_length:].values
    
    scaled_data = scaler.transform(recent_data)
    
    return scaled_data, None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    global model, scaler, config
    
    data = request.json
    commodity = data.get('commodity', 'CL=F')
    
    if not load_resources_for_commodity(commodity):
        return jsonify({'error': f'No model found for {commodity}. Please train first.'}), 500
    
    try:
        sequence_data, error = prepare_prediction_data(commodity)
        if error:
            return jsonify({'error': error}), 400
        
        X = sequence_data.reshape(1, sequence_data.shape[0], sequence_data.shape[1])
        
        direction_prob, magnitude = model.predict(X, verbose=0)
        
        direction_prob = float(direction_prob[0][0])
        magnitude = float(magnitude[0][0])
        
        direction = 'UP' if direction_prob > 0.5 else 'DOWN'
        confidence = direction_prob * 100 if direction == 'UP' else (1 - direction_prob) * 100
        
        abs_magnitude = abs(magnitude)
        if confidence > 70 and abs_magnitude > 3:
            strength = 'STRONG'
            strength_score = min(100, confidence + abs_magnitude * 5)
        elif confidence > 55 and abs_magnitude > 1.5:
            strength = 'MODERATE'
            strength_score = min(80, confidence + abs_magnitude * 3)
        else:
            strength = 'WEAK'
            strength_score = min(50, confidence)
        
        feature_names = config['feature_names']
        category_analysis = calculate_directional_contribution(X, feature_names, commodity, direction_prob)
        
        ticker = yf.Ticker(commodity)
        current_price = ticker.history(period='1d')['Close'].iloc[-1]
        
        target_price = current_price * (1 + magnitude / 100)
        
        return jsonify({
            'commodity': commodity,
            'current_price': round(current_price, 2),
            'direction': direction,
            'confidence': round(confidence, 1),
            'predicted_change_pct': round(magnitude, 2),
            'target_price': round(target_price, 2),
            'strength': strength,
            'strength_score': round(strength_score, 1),
            'prediction_horizon': f"{config['prediction_days']} days",
            'raw_probability': round(direction_prob, 4),
            'category_analysis': category_analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/commodities')
def get_commodities():
    commodities = {
        'Available': {
            'CL=F': 'Crude Oil',
            'NG=F': 'Natural Gas',
            'GC=F': 'Gold'
        }
    }
    return jsonify(commodities)


@app.route('/api/status')
def status():
    return jsonify({
        'model_loaded': model is not None,
        'config': config if config else None
    })


if __name__ == '__main__':
    print("="*60)
    print("LSTM COMMODITY PRICE PREDICTOR")
    print("="*60)
    
    if load_resources():
        print(f"\nModel trained for: {config['commodity']}")
        print(f"Prediction horizon: {config['prediction_days']} days")
        print(f"Sequence length: {config['sequence_length']} days")
        print(f"Number of features: {config['n_features']}")
    else:
        print("\nWARNING: No trained model found!")
        print("Run 'python train.py' first to train the model.")
    
    print("\nStarting web server...")
    print("Open http://localhost:5000 in your browser")
    print("="*60)
    
    app.run(debug=True, port=5000)
