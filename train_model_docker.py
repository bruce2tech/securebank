#!/usr/bin/env python3
"""
Train a model during Docker build to ensure compatibility
"""
import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

print("=" * 60)
print("TRAINING MODEL DURING DOCKER BUILD")
print("=" * 60)

# Check if we have a dataset
dataset_path = None
if Path("storage/datasets/dataset_engineered_raw.csv").exists():
    dataset_path = Path("storage/datasets/dataset_engineered_raw.csv")
    print(f"Found engineered dataset")
elif Path("storage/datasets").exists():
    datasets = sorted(Path("storage/datasets").glob("dataset_*.csv"))
    if datasets:
        dataset_path = datasets[-1]
        print(f"Using dataset: {dataset_path.name}")

if dataset_path and dataset_path.stat().st_size > 0:
    # Load existing dataset
    print(f"\nLoading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Prepare features and target
    if 'is_fraud' in df.columns:
        X = df.drop(['is_fraud'], axis=1, errors='ignore')
        y = df['is_fraud']
    else:
        print("No is_fraud column, creating synthetic labels")
        X = df
        y = (np.random.random(len(df)) < 0.01).astype(int)
    
    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
else:
    # Create synthetic dataset if no data exists
    print("\nNo dataset found, creating synthetic data for initial model")
    np.random.seed(42)
    n_samples = 10000
    n_features = 58  # Match expected feature count
    
    # Create synthetic features with realistic names
    feature_names = ['amt', 'unix_time', 'merch_lat', 'merch_long', 
                     'category_encoded', 'merchant_encoded', 'hour', 
                     'day_of_week', 'is_weekend']
    
    # Add additional feature names
    for i in range(9, n_features):
        feature_names.append(f'feature_{i}')
    
    # Create dataframe
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names[:n_features]
    )
    
    # Create imbalanced target (1% fraud)
    y = pd.Series((np.random.random(n_samples) < 0.01).astype(int))

print(f"\nDataset shape: {X.shape}")
print(f"Fraud rate: {y.mean():.2%}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Try to use ADASYN if available
try:
    from imblearn.over_sampling import ADASYN
    adasyn = ADASYN(sampling_strategy=0.03, random_state=42, n_neighbors=5)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    print(f"ADASYN applied: {len(X_resampled)} samples")
except Exception as e:
    X_resampled, y_resampled = X_train, y_train
    print(f"ADASYN not available: {e}")

# Train model - prefer LightGBM but fall back to RandomForest
model_trained = False

try:
    import lightgbm as lgb
    print("\nTraining LightGBM model...")
    
    # Use EXACT Aggressive configuration that achieved 78.44% precision
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'dart',  # DART proven to work best
        'num_leaves': 70,
        'learning_rate': 0.03,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.5,
        'bagging_freq': 3,
        'min_data_in_leaf': 20,
        'min_gain_to_split': 0.005,
        'lambda_l1': 0.01,
        'lambda_l2': 0.01,
        'scale_pos_weight': 2,
        'drop_rate': 0.1,
        'max_drop': 50,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_resampled, label=y_resampled)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )
    
    # Evaluate
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    model_type = "lightgbm"
    model_trained = True
    
except Exception as e:
    print(f"LightGBM failed: {e}")

if not model_trained:
    print("\nTraining Random Forest as fallback...")
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    model_type = "random_forest"

# Calculate metrics
precision = float(precision_score(y_test, y_pred, zero_division=0))
recall = float(recall_score(y_test, y_pred, zero_division=0))
f1 = float(f1_score(y_test, y_pred, zero_division=0))

print(f"\nModel Performance:")
print(f"  Precision: {precision:.2%}")
print(f"  Recall: {recall:.2%}")
print(f"  F1-Score: {f1:.2%}")

# Save model
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Save in the format app expects
model_data = {
    'model': model,
    'feature_names': list(X.columns),
    'feature_count': len(X.columns),
    'threshold': 0.5,
    'model_type': model_type,
    'performance': {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    },
    'trained_at': datetime.now().isoformat()
}

# Save pickle format
with open("output/best_lgb_model.pkl", 'wb') as f:
    pickle.dump(model_data, f, protocol=2)

# Save text format if LightGBM
if model_type == "lightgbm":
    try:
        model.save_model("output/lightgbm_model.txt")
    except:
        pass

# Save feature names
with open("output/feature_names.json", 'w') as f:
    json.dump({
        'features': list(X.columns),
        'count': len(X.columns)
    }, f, indent=2)

print(f"\n✅ Model trained and saved during Docker build")
print(f"   Type: {model_type}")
print(f"   Features: {len(X.columns)}")
print(f"   Location: output/best_lgb_model.pkl")
print("=" * 60)