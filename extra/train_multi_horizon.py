"""
Multi-Horizon LSTM Training Script
Trains separate models for different prediction timeframes (1d, 5d, 10d, 20d)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from data_fetcher import fetch_all_data, ALL_COMMODITIES
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Prediction horizons to train
HORIZONS = [1, 5, 10, 20]  # days ahead


def prepare_data_for_horizon(df, target_commodity, sequence_length=60, prediction_days=5):
    """Prepare LSTM data for a specific prediction horizon"""
    target_col = f'{target_commodity}_Close'
    
    # Create target: future price change
    future_returns = df[target_col].pct_change(prediction_days).shift(-prediction_days) * 100
    direction = (future_returns > 0).astype(int)
    
    # Get feature columns
    feature_cols = [col for col in df.columns]
    
    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    # Create sequences
    X, y_dir, y_pct = [], [], []
    
    for i in range(sequence_length, len(scaled_data) - prediction_days):
        X.append(scaled_data[i-sequence_length:i])
        y_dir.append(direction.iloc[i])
        y_pct.append(future_returns.iloc[i])
    
    X = np.array(X)
    y_dir = np.array(y_dir)
    y_pct = np.array(y_pct)
    
    # Remove NaN
    valid_idx = ~(np.isnan(y_dir) | np.isnan(y_pct))
    X = X[valid_idx]
    y_dir = y_dir[valid_idx]
    y_pct = y_pct[valid_idx]
    
    return X, y_dir, y_pct, feature_cols, scaler


def create_model(seq_len, n_features, dropout=0.3, l2_reg=0.001):
    """Create LSTM model"""
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
    """Evaluate model performance"""
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


def train_for_horizon(df, commodity, horizon, sequence_length=60, epochs=150):
    """Train a model for a specific prediction horizon"""
    print(f"\n{'='*60}")
    print(f"TRAINING {horizon}-DAY PREDICTION MODEL")
    print(f"{'='*60}")
    
    # Prepare data
    X, y_dir, y_pct, feature_names, scaler = prepare_data_for_horizon(
        df, commodity, sequence_length, horizon
    )
    
    print(f"Data shape: {X.shape}")
    print(f"Samples: {len(X)}")
    
    # Split data (time-based, no shuffle)
    X_trainval, X_test, y_dir_trainval, y_dir_test, y_pct_trainval, y_pct_test = train_test_split(
        X, y_dir, y_pct, test_size=0.15, shuffle=False
    )
    X_train, X_val, y_dir_train, y_dir_val, y_pct_train, y_pct_val = train_test_split(
        X_trainval, y_dir_trainval, y_pct_trainval, test_size=0.18, shuffle=False
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create and train model
    model = create_model(X.shape[1], X.shape[2])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0)
    ]
    
    history = model.fit(
        X_train, {'direction': y_dir_train, 'magnitude': y_pct_train},
        validation_data=(X_val, {'direction': y_dir_val, 'magnitude': y_pct_val}),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    
    epochs_trained = len(history.history['loss'])
    print(f"Epochs trained: {epochs_trained}")
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_dir_test, y_pct_test)
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"AUC-ROC:  {metrics['auc_roc']:.4f}")
    print(f"F1 Score: {metrics['f1_score']*100:.2f}%")
    print(f"MAE:      {metrics['mae']:.2f}%")
    
    return model, scaler, feature_names, metrics


def main():
    parser = argparse.ArgumentParser(description='Train multi-horizon LSTM models')
    parser.add_argument('--commodity', type=str, default='CL=F', help='Commodity symbol')
    parser.add_argument('--years', type=int, default=5, help='Years of historical data')
    parser.add_argument('--sequence', type=int, default=60, help='Sequence length (lookback days)')
    parser.add_argument('--epochs', type=int, default=150, help='Max epochs')
    args = parser.parse_args()
    
    print("="*60)
    print("MULTI-HORIZON LSTM TRAINING")
    print("="*60)
    print(f"Commodity: {args.commodity}")
    print(f"Data: {args.years} years")
    print(f"Lookback: {args.sequence} days")
    print(f"Horizons: {HORIZONS} days")
    
    # Fetch data once
    print("\n[1] Fetching data...")
    df, feature_categories = fetch_all_data(args.commodity, years=args.years)
    
    # Get date range
    date_start = df.index[0].strftime('%Y-%m-%d')
    date_end = df.index[-1].strftime('%Y-%m-%d')
    print(f"\nData range: {date_start} to {date_end}")
    print(f"Total data points: {len(df)}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train model for each horizon
    results = {}
    
    for horizon in HORIZONS:
        model, scaler, feature_names, metrics = train_for_horizon(
            df, args.commodity, horizon, args.sequence, args.epochs
        )
        
        # Save model and config for this horizon
        model_path = f'models/lstm_{horizon}d.keras'
        scaler_path = f'models/scaler_{horizon}d.joblib'
        config_path = f'models/config_{horizon}d.joblib'
        
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump({
            'commodity': args.commodity,
            'commodity_name': ALL_COMMODITIES.get(args.commodity, args.commodity),
            'sequence_length': args.sequence,
            'prediction_days': horizon,
            'feature_names': feature_names,
            'feature_categories': feature_categories,
            'n_features': len(feature_names),
            'metrics': metrics,
            'data_start': date_start,
            'data_end': date_end,
            'data_points': len(df),
            'trained_at': datetime.now().isoformat()
        }, config_path)
        
        results[horizon] = metrics
        print(f"Saved: {model_path}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"\n{'Horizon':<10} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>10} {'MAE':>10}")
    print("-"*50)
    for horizon in HORIZONS:
        m = results[horizon]
        print(f"{horizon} days{'':<5} {m['accuracy']*100:>9.2f}% {m['auc_roc']:>10.4f} {m['f1_score']*100:>9.2f}% {m['mae']:>9.2f}%")
    
    # Save master config
    joblib.dump({
        'commodity': args.commodity,
        'commodity_name': ALL_COMMODITIES.get(args.commodity, args.commodity),
        'horizons': HORIZONS,
        'sequence_length': args.sequence,
        'data_start': date_start,
        'data_end': date_end,
        'data_points': len(df),
        'results': results
    }, 'models/master_config.joblib')
    
    print(f"\nAll models saved to models/")
    print(f"Training data: {date_start} to {date_end} ({len(df)} data points)")


if __name__ == "__main__":
    main()
