import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import joblib
import numpy as np
from datetime import datetime
from data_fetcher import fetch_all_data, prepare_lstm_data
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

COMMODITIES = {
    'CL=F': 'Crude Oil',
    'NG=F': 'Natural Gas',
    'GC=F': 'Gold'
}


def create_model(seq_len, n_features, dropout=0.3, l2_reg=0.001):
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
    x = Dense(32, activation='relu')(x)
    
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
    dir_probs, mag_preds = model.predict(X_test, verbose=0)
    dir_probs = dir_probs.flatten()
    dir_preds = (dir_probs > 0.5).astype(int)
    y_true = y_dir_test.flatten()
    
    return {
        'accuracy': float(accuracy_score(y_true, dir_preds)),
        'f1_score': float(f1_score(y_true, dir_preds, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_true, dir_probs)),
        'brier': float(brier_score_loss(y_true, dir_probs)),
        'mae': float(np.mean(np.abs(y_pct_test - mag_preds.flatten())))
    }


def train_commodity(symbol, years=5, sequence_length=60, prediction_days=5, epochs=100):
    name = COMMODITIES.get(symbol, symbol)
    
    print("\n" + "="*60)
    print(f"TRAINING MODEL FOR: {name} ({symbol})")
    print("="*60)
    
    print("\n[1/4] Fetching data...")
    df, feature_categories = fetch_all_data(symbol, years=years)
    
    date_start = df.index[0].strftime('%Y-%m-%d')
    date_end = df.index[-1].strftime('%Y-%m-%d')
    print(f"Data range: {date_start} to {date_end}")
    print(f"Data points: {len(df)}")
    
    print("\n[2/4] Preparing sequences...")
    X, y_dir, y_pct, feature_names, scaler = prepare_lstm_data(
        df, target_commodity=symbol, 
        sequence_length=sequence_length, 
        prediction_days=prediction_days
    )
    
    X_trainval, X_test, y_dir_trainval, y_dir_test, y_pct_trainval, y_pct_test = train_test_split(
        X, y_dir, y_pct, test_size=0.15, shuffle=False
    )
    X_train, X_val, y_dir_train, y_dir_val, y_pct_train, y_pct_val = train_test_split(
        X_trainval, y_dir_trainval, y_pct_trainval, test_size=0.18, shuffle=False
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    print("\n[3/4] Training model...")
    model = create_model(X.shape[1], X.shape[2])
    print(f"Model parameters: {model.count_params():,}")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    history = model.fit(
        X_train, {'direction': y_dir_train, 'magnitude': y_pct_train},
        validation_data=(X_val, {'direction': y_dir_val, 'magnitude': y_pct_val}),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    epochs_trained = len(history.history['loss'])
    
    print("\n[4/4] Evaluating...")
    metrics = evaluate_model(model, X_test, y_dir_test, y_pct_test)
    
    print(f"\n{'='*40}")
    print("RESULTS")
    print(f"{'='*40}")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"AUC-ROC:  {metrics['auc_roc']:.4f}")
    print(f"F1 Score: {metrics['f1_score']*100:.2f}%")
    print(f"MAE:      {metrics['mae']:.2f}%")
    
    os.makedirs('models', exist_ok=True)
    
    safe_symbol = symbol.replace('=', '_')
    
    model_path = f'models/{safe_symbol}_model.keras'
    scaler_path = f'models/{safe_symbol}_scaler.joblib'
    config_path = f'models/{safe_symbol}_config.joblib'
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump({
        'symbol': symbol,
        'name': name,
        'sequence_length': sequence_length,
        'prediction_days': prediction_days,
        'feature_names': feature_names,
        'feature_categories': feature_categories,
        'n_features': len(feature_names),
        'metrics': metrics,
        'data_start': date_start,
        'data_end': date_end,
        'data_points': len(df),
        'epochs_trained': epochs_trained,
        'trained_at': datetime.now().isoformat()
    }, config_path)
    
    print(f"\nSaved: {model_path}")
    print(f"Saved: {scaler_path}")
    print(f"Saved: {config_path}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description='Train LSTM model for commodity prediction')
    parser.add_argument('--commodity', type=str, default='all', 
                        help='Commodity symbol (CL=F, NG=F, GC=F) or "all"')
    parser.add_argument('--years', type=int, default=5, help='Years of data')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    args = parser.parse_args()
    
    if args.commodity == 'all':
        results = {}
        for symbol in COMMODITIES:
            _, metrics = train_commodity(symbol, years=args.years, epochs=args.epochs)
            results[symbol] = metrics
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"\n{'Commodity':<20} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>10}")
        print("-"*50)
        for symbol, m in results.items():
            name = COMMODITIES[symbol]
            print(f"{name:<20} {m['accuracy']*100:>9.2f}% {m['auc_roc']:>10.4f} {m['f1_score']*100:>9.2f}%")
    else:
        if args.commodity not in COMMODITIES:
            print(f"Unknown commodity: {args.commodity}")
            print(f"Available: {list(COMMODITIES.keys())}")
            return
        train_commodity(args.commodity, years=args.years, epochs=args.epochs)


if __name__ == "__main__":
    main()
