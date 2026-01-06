"""
Model Comparison Script
Compares multiple architectures with various anti-overfitting techniques
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Input, Bidirectional, BatchNormalization,
    Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D,
    MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, accuracy_score
import joblib
from data_fetcher import fetch_all_data, prepare_lstm_data

# ============================================================
# ANTI-OVERFITTING TECHNIQUES USED:
# ============================================================
# 1. Dropout (0.3-0.5) - Randomly drops neurons during training
# 2. L2 Regularization - Penalizes large weights
# 3. Batch Normalization - Normalizes layer outputs
# 4. Early Stopping - Stops when validation loss stops improving
# 5. Learning Rate Reduction - Reduces LR on plateau
# 6. Smaller model capacity - Less parameters = less overfitting
# 7. Data augmentation via noise injection
# ============================================================


def add_noise_augmentation(X, noise_factor=0.01):
    """Add small random noise to training data for augmentation"""
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise


def create_lstm_model(seq_len, n_features, dropout=0.3, l2_reg=0.001):
    """Standard LSTM with anti-overfitting"""
    inputs = Input(shape=(seq_len, n_features))
    
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    
    x = LSTM(32, return_sequences=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout * 0.5)(x)
    
    direction = Dense(1, activation='sigmoid', name='direction')(x)
    magnitude = Dense(1, activation='linear', name='magnitude')(x)
    
    model = Model(inputs=inputs, outputs=[direction, magnitude])
    return model


def create_gru_model(seq_len, n_features, dropout=0.3, l2_reg=0.001):
    """GRU model - faster than LSTM, often similar performance"""
    inputs = Input(shape=(seq_len, n_features))
    
    x = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    
    x = GRU(32, return_sequences=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout * 0.5)(x)
    
    direction = Dense(1, activation='sigmoid', name='direction')(x)
    magnitude = Dense(1, activation='linear', name='magnitude')(x)
    
    model = Model(inputs=inputs, outputs=[direction, magnitude])
    return model


def create_cnn_lstm_model(seq_len, n_features, dropout=0.3, l2_reg=0.001):
    """CNN + LSTM hybrid - CNN extracts local patterns, LSTM captures sequence"""
    inputs = Input(shape=(seq_len, n_features))
    
    # CNN layers for local pattern extraction
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout)(x)
    
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    
    # LSTM for sequence modeling
    x = LSTM(32, return_sequences=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout * 0.5)(x)
    
    direction = Dense(1, activation='sigmoid', name='direction')(x)
    magnitude = Dense(1, activation='linear', name='magnitude')(x)
    
    model = Model(inputs=inputs, outputs=[direction, magnitude])
    return model


def create_transformer_model(seq_len, n_features, dropout=0.3, l2_reg=0.001):
    """Transformer-based model using self-attention"""
    inputs = Input(shape=(seq_len, n_features))
    
    # Project to embedding dimension
    x = Dense(64, kernel_regularizer=l2(l2_reg))(inputs)
    x = LayerNormalization()(x)
    
    # Multi-head attention
    attn_output = MultiHeadAttention(num_heads=4, key_dim=16, dropout=dropout)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)
    
    # Feed-forward
    ff = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    ff = Dropout(dropout)(ff)
    ff = Dense(64, kernel_regularizer=l2(l2_reg))(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout)(x)
    
    direction = Dense(1, activation='sigmoid', name='direction')(x)
    magnitude = Dense(1, activation='linear', name='magnitude')(x)
    
    model = Model(inputs=inputs, outputs=[direction, magnitude])
    return model


def create_simple_dense_model(seq_len, n_features, dropout=0.4, l2_reg=0.01):
    """Simple Dense network - baseline comparison"""
    inputs = Input(shape=(seq_len, n_features))
    
    x = Flatten()(inputs)
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout)(x)
    
    direction = Dense(1, activation='sigmoid', name='direction')(x)
    magnitude = Dense(1, activation='linear', name='magnitude')(x)
    
    model = Model(inputs=inputs, outputs=[direction, magnitude])
    return model


def compile_model(model, learning_rate=0.001):
    """Compile model with standard settings"""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={'direction': 'binary_crossentropy', 'magnitude': 'huber'},
        loss_weights={'direction': 1.0, 'magnitude': 0.3},
        metrics={'direction': 'accuracy', 'magnitude': 'mae'}
    )
    return model


def evaluate_model(model, X_test, y_dir_test, y_pct_test):
    """Calculate metrics"""
    direction_probs, magnitude_preds = model.predict(X_test, verbose=0)
    direction_probs = direction_probs.flatten()
    magnitude_preds = magnitude_preds.flatten()
    direction_preds = (direction_probs > 0.5).astype(int)
    y_true = y_dir_test.flatten()
    
    return {
        'accuracy': accuracy_score(y_true, direction_preds),
        'f1_score': f1_score(y_true, direction_preds, zero_division=0),
        'auc_roc': roc_auc_score(y_true, direction_probs),
        'brier': brier_score_loss(y_true, direction_probs),
        'mae': np.mean(np.abs(y_pct_test - magnitude_preds))
    }


def train_and_evaluate(model_fn, model_name, X_train, X_val, X_test, 
                       y_dir_train, y_dir_val, y_dir_test,
                       y_pct_train, y_pct_val, y_pct_test,
                       epochs=150, use_augmentation=False):
    """Train a model and return metrics"""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    seq_len, n_features = X_train.shape[1], X_train.shape[2]
    model = model_fn(seq_len, n_features)
    model = compile_model(model)
    
    # Print param count
    total_params = model.count_params()
    print(f"Parameters: {total_params:,}")
    
    # Data augmentation
    if use_augmentation:
        X_train_aug = add_noise_augmentation(X_train, noise_factor=0.005)
        X_train_combined = np.concatenate([X_train, X_train_aug])
        y_dir_combined = np.concatenate([y_dir_train, y_dir_train])
        y_pct_combined = np.concatenate([y_pct_train, y_pct_train])
        print(f"Augmented training samples: {len(X_train_combined)}")
    else:
        X_train_combined = X_train
        y_dir_combined = y_dir_train
        y_pct_combined = y_pct_train
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0)
    ]
    
    history = model.fit(
        X_train_combined,
        {'direction': y_dir_combined, 'magnitude': y_pct_combined},
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
    
    print(f"Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Test AUC-ROC:  {metrics['auc_roc']:.4f}")
    print(f"Test F1:       {metrics['f1_score']*100:.2f}%")
    print(f"Brier Score:   {metrics['brier']:.4f}")
    print(f"MAE:           {metrics['mae']:.3f}%")
    
    return model, metrics, epochs_trained


def main():
    print("="*60)
    print("MODEL COMPARISON WITH ANTI-OVERFITTING TECHNIQUES")
    print("="*60)
    
    print("\n[Anti-Overfitting Techniques Applied]")
    print("1. Dropout (0.3-0.5)")
    print("2. L2 Regularization (weight decay)")
    print("3. Batch Normalization")
    print("4. Early Stopping (patience=20)")
    print("5. Learning Rate Reduction on Plateau")
    print("6. Reduced model capacity")
    print("7. Data augmentation (noise injection)")
    
    # Load data
    print("\n[Loading Data...]")
    config = joblib.load('models/config.joblib')
    df, _ = fetch_all_data(config['commodity'], years=5)
    
    X, y_direction, y_pct_change, feature_names, scaler = prepare_lstm_data(
        df, 
        target_commodity=config['commodity'],
        sequence_length=config['sequence_length'],
        prediction_days=config['prediction_days']
    )
    
    # Split: 70% train, 15% val, 15% test
    X_trainval, X_test, y_dir_trainval, y_dir_test, y_pct_trainval, y_pct_test = train_test_split(
        X, y_direction, y_pct_change, test_size=0.15, shuffle=False
    )
    X_train, X_val, y_dir_train, y_dir_val, y_pct_train, y_pct_val = train_test_split(
        X_trainval, y_dir_trainval, y_pct_trainval, test_size=0.18, shuffle=False
    )
    
    print(f"\nData split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Models to compare
    models_to_test = [
        (create_lstm_model, "LSTM (Bidirectional)", False),
        (create_lstm_model, "LSTM + Augmentation", True),
        (create_gru_model, "GRU (Bidirectional)", False),
        (create_cnn_lstm_model, "CNN-LSTM Hybrid", False),
        (create_transformer_model, "Transformer", False),
        (create_simple_dense_model, "Dense Baseline", False),
    ]
    
    results = []
    
    for model_fn, name, use_aug in models_to_test:
        try:
            model, metrics, epochs = train_and_evaluate(
                model_fn, name,
                X_train, X_val, X_test,
                y_dir_train, y_dir_val, y_dir_test,
                y_pct_train, y_pct_val, y_pct_test,
                epochs=150,
                use_augmentation=use_aug
            )
            results.append({
                'name': name,
                'model': model,
                'metrics': metrics,
                'epochs': epochs
            })
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Model':<25} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>10} {'Brier':>10} {'MAE':>10}")
    print("-"*75)
    
    best_auc = 0
    best_model_info = None
    
    for r in results:
        m = r['metrics']
        print(f"{r['name']:<25} {m['accuracy']*100:>9.2f}% {m['auc_roc']:>10.4f} {m['f1_score']*100:>9.2f}% {m['brier']:>10.4f} {m['mae']:>9.3f}%")
        
        if m['auc_roc'] > best_auc:
            best_auc = m['auc_roc']
            best_model_info = r
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_info['name']}")
    print(f"AUC-ROC: {best_model_info['metrics']['auc_roc']:.4f}")
    print("="*60)
    
    # Save best model
    if best_model_info:
        best_model_info['model'].save('models/best_model.keras')
        print(f"\nBest model saved to models/best_model.keras")
    
    return results


if __name__ == "__main__":
    results = main()
