"""Quick evaluation script for LSTM model"""
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import tensorflow as tf
from data_fetcher import fetch_all_data, prepare_lstm_data

print("="*60)
print("LSTM MODEL EVALUATION")
print("="*60)

# Load model and config
model = tf.keras.models.load_model('models/lstm_commodity_model.keras')
scaler = joblib.load('models/scaler.joblib')
config = joblib.load('models/config.joblib')

print(f"Commodity: {config['commodity']}")
print(f"Features: {config['n_features']}")
print(f"Sequence Length: {config['sequence_length']} days")
print(f"Prediction Horizon: {config['prediction_days']} days")

# Fetch fresh data and prepare test set
print("\nFetching data for evaluation...")
df, _ = fetch_all_data(config['commodity'], years=5)

X, y_direction, y_pct_change, feature_names, _ = prepare_lstm_data(
    df, 
    target_commodity=config['commodity'],
    sequence_length=config['sequence_length'],
    prediction_days=config['prediction_days']
)

# Use last 20% as test set
test_size = int(len(X) * 0.2)
X_test = X[-test_size:]
y_dir_test = y_direction[-test_size:]
y_pct_test = y_pct_change[-test_size:]

print(f"Test samples: {len(X_test)}")

# Get predictions
print("\nRunning predictions...")
direction_probs, magnitude_preds = model.predict(X_test, verbose=0)
direction_probs = direction_probs.flatten()
magnitude_preds = magnitude_preds.flatten()

# Convert to binary
direction_preds = (direction_probs > 0.5).astype(int)
y_true = y_dir_test.flatten()

# Calculate metrics
print("\n" + "="*60)
print("PERFORMANCE METRICS")
print("="*60)

accuracy = accuracy_score(y_true, direction_preds)
precision = precision_score(y_true, direction_preds, zero_division=0)
recall = recall_score(y_true, direction_preds, zero_division=0)
f1 = f1_score(y_true, direction_preds, zero_division=0)
auc = roc_auc_score(y_true, direction_probs)
brier = brier_score_loss(y_true, direction_probs)

print("\n--- Direction Prediction (Classification) ---")
print(f"Accuracy:     {accuracy*100:.2f}%")
print(f"Precision:    {precision*100:.2f}%")
print(f"Recall:       {recall*100:.2f}%")
print(f"F1 Score:     {f1*100:.2f}%")
print(f"AUC-ROC:      {auc:.4f}")
print(f"Brier Score:  {brier:.4f} (lower is better, 0 = perfect)")

# Magnitude metrics
mae = np.mean(np.abs(y_pct_test - magnitude_preds))
rmse = np.sqrt(np.mean((y_pct_test - magnitude_preds) ** 2))
mape = np.mean(np.abs((y_pct_test - magnitude_preds) / (np.abs(y_pct_test) + 0.001))) * 100

print("\n--- Magnitude Prediction (Regression) ---")
print(f"MAE:          {mae:.3f}%")
print(f"RMSE:         {rmse:.3f}%")
print(f"MAPE:         {mape:.2f}%")

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, direction_preds).ravel()
print("\n--- Confusion Matrix ---")
print(f"True Positives (UP correct):   {tp}")
print(f"True Negatives (DOWN correct): {tn}")
print(f"False Positives (UP wrong):    {fp}")
print(f"False Negatives (DOWN wrong):  {fn}")

# Class distribution
print("\n--- Data Distribution ---")
print(f"Actual UP:   {y_true.sum()} ({y_true.sum()/len(y_true)*100:.1f}%)")
print(f"Actual DOWN: {len(y_true) - y_true.sum()} ({(len(y_true) - y_true.sum())/len(y_true)*100:.1f}%)")

# Additional insight
print("\n--- Summary ---")
if auc > 0.6:
    print(f"Model has predictive power (AUC > 0.5 baseline)")
else:
    print(f"Model performance similar to random (AUC near 0.5)")
    
if brier < 0.25:
    print(f"Probability calibration is reasonable")
