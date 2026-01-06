"""
Training Script for LSTM Commodity Price Predictor
Run this to fetch data and train the model
"""

import os
import argparse
import joblib
import numpy as np
from data_fetcher import fetch_all_data, prepare_lstm_data, ALL_COMMODITIES
from lstm_model import train_model
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, f1_score, precision_score, 
    recall_score, accuracy_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_direction_test, y_pct_test):
    """Calculate comprehensive model performance metrics"""
    # Get predictions
    direction_probs, magnitude_preds = model.predict(X_test, verbose=0)
    direction_probs = direction_probs.flatten()
    magnitude_preds = magnitude_preds.flatten()
    
    # Convert probabilities to binary predictions
    direction_preds = (direction_probs > 0.5).astype(int)
    y_true = y_direction_test.flatten()
    
    # Classification metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, direction_preds)),
        'precision': float(precision_score(y_true, direction_preds, zero_division=0)),
        'recall': float(recall_score(y_true, direction_preds, zero_division=0)),
        'f1_score': float(f1_score(y_true, direction_preds, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_true, direction_probs)),
        'brier_score': float(brier_score_loss(y_true, direction_probs)),
    }
    
    # Magnitude metrics
    mae = float(np.mean(np.abs(y_pct_test - magnitude_preds)))
    rmse = float(np.sqrt(np.mean((y_pct_test - magnitude_preds) ** 2)))
    metrics['magnitude_mae'] = mae
    metrics['magnitude_rmse'] = rmse
    
    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, direction_preds).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    # Additional useful metrics
    metrics['total_samples'] = int(len(y_true))
    metrics['up_predictions'] = int(direction_preds.sum())
    metrics['down_predictions'] = int(len(direction_preds) - direction_preds.sum())
    metrics['actual_up'] = int(y_true.sum())
    metrics['actual_down'] = int(len(y_true) - y_true.sum())
    
    return metrics


def print_metrics(metrics):
    """Print formatted metrics"""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    
    print("\n📊 Classification Metrics (Direction Prediction):")
    print(f"  Accuracy:     {metrics['accuracy']*100:.2f}%")
    print(f"  Precision:    {metrics['precision']*100:.2f}%")
    print(f"  Recall:       {metrics['recall']*100:.2f}%")
    print(f"  F1 Score:     {metrics['f1_score']*100:.2f}%")
    print(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"  Brier Score:  {metrics['brier_score']:.4f} (lower is better)")
    
    print("\n📈 Magnitude Metrics (Price Change %):")
    print(f"  MAE:          {metrics['magnitude_mae']:.2f}%")
    print(f"  RMSE:         {metrics['magnitude_rmse']:.2f}%")
    
    print("\n📋 Confusion Matrix:")
    print(f"  True Positives (UP correct):   {metrics['true_positives']}")
    print(f"  True Negatives (DOWN correct): {metrics['true_negatives']}")
    print(f"  False Positives (UP wrong):    {metrics['false_positives']}")
    print(f"  False Negatives (DOWN wrong):  {metrics['false_negatives']}")
    
    print("\n📉 Distribution:")
    print(f"  Actual UP:   {metrics['actual_up']} ({metrics['actual_up']/metrics['total_samples']*100:.1f}%)")
    print(f"  Actual DOWN: {metrics['actual_down']} ({metrics['actual_down']/metrics['total_samples']*100:.1f}%)")


def plot_training_history(history, save_path='models/'):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Direction Accuracy
    axes[0, 1].plot(history.history['direction_accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_direction_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Direction Prediction Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Direction Loss
    axes[1, 0].plot(history.history['direction_loss'], label='Train')
    axes[1, 0].plot(history.history['val_direction_loss'], label='Val')
    axes[1, 0].set_title('Direction Loss (Binary Crossentropy)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Magnitude MAE
    axes[1, 1].plot(history.history['magnitude_mae'], label='Train MAE')
    axes[1, 1].plot(history.history['val_magnitude_mae'], label='Val MAE')
    axes[1, 1].set_title('Magnitude Prediction MAE (%)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=150)
    plt.close()
    print(f"Training history plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train LSTM Commodity Price Predictor')
    parser.add_argument('--commodity', type=str, default='CL=F', 
                        help='Target commodity symbol (default: CL=F for Crude Oil)')
    parser.add_argument('--years', type=int, default=5,
                        help='Years of historical data (default: 5)')
    parser.add_argument('--sequence', type=int, default=60,
                        help='Sequence length for LSTM (default: 60 days)')
    parser.add_argument('--predict-days', type=int, default=5,
                        help='Days ahead to predict (default: 5)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LSTM COMMODITY PRICE PREDICTOR - TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Target Commodity: {args.commodity}")
    print(f"  Historical Data: {args.years} years")
    print(f"  Sequence Length: {args.sequence} days")
    print(f"  Prediction Horizon: {args.predict_days} days")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print("\n" + "="*60)
    
    # Step 1: Fetch data
    print("\n[1/3] Fetching data...")
    df, feature_categories = fetch_all_data(args.commodity, years=args.years)
    
    # Step 2: Prepare LSTM data
    print("\n[2/3] Preparing LSTM sequences...")
    X, y_direction, y_pct_change, feature_names, scaler = prepare_lstm_data(
        df, 
        target_commodity=args.commodity,
        sequence_length=args.sequence,
        prediction_days=args.predict_days
    )
    
    # Save scaler and config
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump({
        'commodity': args.commodity,
        'sequence_length': args.sequence,
        'prediction_days': args.predict_days,
        'feature_names': feature_names,
        'feature_categories': feature_categories,
        'n_features': X.shape[2]
    }, 'models/config.joblib')
    print("Scaler and config saved.")
    
    # Step 3: Train model
    print("\n[3/3] Training LSTM model...")
    model, history, X_test, y_dir_test, y_pct_test = train_model(
        X, y_direction, y_pct_change,
        model_save_path='models/',
        epochs=args.epochs,
        batch_size=args.batch_size,
        return_test_data=True
    )
    
    # Plot training history
    plot_training_history(history, 'models/')
    
    # Step 4: Evaluate model on test set
    print("\n[4/4] Evaluating model performance...")
    metrics = evaluate_model(model, X_test, y_dir_test, y_pct_test)
    print_metrics(metrics)
    
    # Update config with metrics
    config = joblib.load('models/config.joblib')
    config['metrics'] = metrics
    config['training_epochs'] = args.epochs
    config['final_val_loss'] = float(history.history['val_loss'][-1])
    config['final_val_accuracy'] = float(history.history['val_direction_accuracy'][-1])
    joblib.dump(config, 'models/config.joblib')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: models/lstm_commodity_model.keras")
    print(f"Scaler saved to: models/scaler.joblib")
    print(f"Config saved to: models/config.joblib (with metrics)")
    print(f"\nTo make predictions, run: python app.py")
    
    # Show available commodities
    print("\n" + "-"*60)
    print("Available commodities for training:")
    for symbol, name in ALL_COMMODITIES.items():
        print(f"  {symbol}: {name}")


if __name__ == "__main__":
    main()
