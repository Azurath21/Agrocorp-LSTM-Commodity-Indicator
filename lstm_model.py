"""
LSTM Model for Commodity Price Prediction
Predicts price direction (up/down) and strength of movement
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import joblib
import os


def create_lstm_model(sequence_length, n_features, model_type='direction'):
    """
    Create LSTM model for price prediction
    
    Args:
        sequence_length: Number of time steps
        n_features: Number of input features
        model_type: 'direction' for classification, 'magnitude' for regression
    """
    model = Sequential([
        # First LSTM layer with return sequences
        Bidirectional(LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features))),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second LSTM layer with return sequences
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third LSTM layer
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        
        # Dense layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
    ])
    
    if model_type == 'direction':
        # Binary classification: up or down
        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        # Regression: predict percentage change
        model.add(Dense(1, activation='linear'))
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )
    
    return model


def create_multi_output_model(sequence_length, n_features):
    """
    Create model that predicts both direction and magnitude
    """
    inputs = Input(shape=(sequence_length, n_features))
    
    # Shared LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(32, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Shared dense layer
    shared = Dense(64, activation='relu')(x)
    shared = BatchNormalization()(shared)
    shared = Dropout(0.2)(shared)
    
    # Direction branch (classification)
    direction_branch = Dense(32, activation='relu')(shared)
    direction_output = Dense(1, activation='sigmoid', name='direction')(direction_branch)
    
    # Magnitude branch (regression)
    magnitude_branch = Dense(32, activation='relu')(shared)
    magnitude_output = Dense(1, activation='linear', name='magnitude')(magnitude_branch)
    
    model = Model(inputs=inputs, outputs=[direction_output, magnitude_output])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'direction': 'binary_crossentropy',
            'magnitude': 'huber'
        },
        loss_weights={
            'direction': 1.0,
            'magnitude': 0.5
        },
        metrics={
            'direction': 'accuracy',
            'magnitude': 'mae'
        }
    )
    
    return model


def train_model(X, y_direction, y_magnitude, model_save_path='models/', epochs=100, batch_size=32, return_test_data=False):
    """
    Train the LSTM model
    
    Returns:
        model: trained model
        history: training history
        (optional) X_test, y_dir_test, y_mag_test: test data for evaluation
    """
    os.makedirs(model_save_path, exist_ok=True)
    
    # Split data - first split into train+val and test
    X_trainval, X_test, y_dir_trainval, y_dir_test, y_mag_trainval, y_mag_test = train_test_split(
        X, y_direction, y_magnitude, test_size=0.15, shuffle=False
    )
    
    # Then split train+val into train and val
    X_train, X_val, y_dir_train, y_dir_val, y_mag_train, y_mag_val = train_test_split(
        X_trainval, y_dir_trainval, y_mag_trainval, test_size=0.18, shuffle=False
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    sequence_length = X.shape[1]
    n_features = X.shape[2]
    model = create_multi_output_model(sequence_length, n_features)
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(model_save_path, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train,
        {'direction': y_dir_train, 'magnitude': y_mag_train},
        validation_data=(X_val, {'direction': y_dir_val, 'magnitude': y_mag_val}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*50)
    print("Final Evaluation:")
    eval_results = model.evaluate(X_val, {'direction': y_dir_val, 'magnitude': y_mag_val})
    print(f"Validation Loss: {eval_results[0]:.4f}")
    print(f"Direction Accuracy: {eval_results[3]*100:.2f}%")
    print(f"Magnitude MAE: {eval_results[4]:.4f}%")
    
    # Save model
    model.save(os.path.join(model_save_path, 'lstm_commodity_model.keras'))
    print(f"\nModel saved to {model_save_path}")
    
    return model, history


def load_model(model_path='models/lstm_commodity_model.keras'):
    """Load a trained model"""
    return tf.keras.models.load_model(model_path)


def predict(model, X_sequence, scaler=None):
    """
    Make prediction for a single sequence
    
    Args:
        model: trained LSTM model
        X_sequence: input sequence of shape (sequence_length, n_features)
        scaler: StandardScaler used during training
    
    Returns:
        direction: 'UP' or 'DOWN'
        confidence: probability (0-100%)
        magnitude: predicted percentage change
        strength: 'STRONG', 'MODERATE', or 'WEAK'
    """
    # Reshape if needed
    if len(X_sequence.shape) == 2:
        X_sequence = X_sequence.reshape(1, X_sequence.shape[0], X_sequence.shape[1])
    
    # Predict
    direction_prob, magnitude = model.predict(X_sequence, verbose=0)
    
    direction_prob = direction_prob[0][0]
    magnitude = magnitude[0][0]
    
    # Interpret results
    direction = 'UP' if direction_prob > 0.5 else 'DOWN'
    confidence = direction_prob * 100 if direction == 'UP' else (1 - direction_prob) * 100
    
    # Determine strength based on confidence and magnitude
    abs_magnitude = abs(magnitude)
    if confidence > 70 and abs_magnitude > 3:
        strength = 'STRONG'
    elif confidence > 55 and abs_magnitude > 1.5:
        strength = 'MODERATE'
    else:
        strength = 'WEAK'
    
    return {
        'direction': direction,
        'confidence': round(confidence, 2),
        'predicted_change': round(magnitude, 2),
        'strength': strength,
        'raw_probability': round(direction_prob, 4)
    }


if __name__ == "__main__":
    # Test model creation
    model = create_multi_output_model(sequence_length=60, n_features=50)
    model.summary()
