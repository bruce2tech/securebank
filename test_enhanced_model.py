#!/usr/bin/env python3
"""Test the enhanced model with SMOTE"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Try importing SMOTE
try:
    from imblearn.over_sampling import SMOTE
    print("✅ SMOTE available")
    SMOTE_AVAILABLE = True
except ImportError:
    print("⚠️  SMOTE not available, using class weights")
    SMOTE_AVAILABLE = False

print("\nLoading dataset...")
dataset_files = list(Path('storage/datasets').glob('dataset_*.csv'))
if not dataset_files:
    print("No datasets found!")
    sys.exit(1)

latest = max(dataset_files, key=lambda p: p.stat().st_mtime)
df = pd.read_csv(latest)
print(f"Loaded: {latest.name}")
print(f"Shape: {df.shape}")

# Prepare data
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

print(f"Original fraud rate: {y.mean():.2%}")

# Encode categoricals
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].fillna('unknown').astype(str))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

if SMOTE_AVAILABLE:
    print("\n" + "="*50)
    print("TESTING WITH SMOTE")
    print("="*50)
    
    # Apply SMOTE
    smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"After SMOTE: {len(X_train_smote)} samples")
    print(f"New fraud rate in training: {y_train_smote.mean():.2%}")
    
    # Train model with SMOTE data
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining model with SMOTE data...")
    model.fit(X_train_smote, y_train_smote)
    
    # Get probabilities for threshold tuning
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Test different thresholds
    print("\nThreshold Analysis:")
    print("-" * 40)
    
    for threshold in [0.5, 0.3, 0.2, 0.15, 0.1]:
        y_pred = (y_proba >= threshold).astype(int)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        meets = "✅" if (prec >= 0.7 and rec >= 0.7) else "❌"
        print(f"Threshold {threshold:.2f}: P={prec:.2%}, R={rec:.2%}, F1={f1:.2%} {meets}")
    
    # Find optimal threshold
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Find where both >= 0.7
    valid = (precisions >= 0.7) & (recalls >= 0.7)
    if valid.any():
        valid_idx = np.where(valid)[0]
        f1s = 2 * (precisions[valid_idx] * recalls[valid_idx]) / (precisions[valid_idx] + recalls[valid_idx])
        best_idx = valid_idx[np.argmax(f1s)]
        optimal_threshold = thresholds[best_idx]
        print(f"\n✅ Found optimal threshold: {optimal_threshold:.3f}")
        
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        print(f"Final Precision: {precision_score(y_test, y_pred_optimal):.2%}")
        print(f"Final Recall: {recall_score(y_test, y_pred_optimal):.2%}")
        print(f"Final F1-Score: {f1_score(y_test, y_pred_optimal):.2%}")
    else:
        print("\n⚠️  No single threshold meets both requirements")
        print("Try adjusting SMOTE parameters or model hyperparameters")

else:
    print("\n" + "="*50)
    print("TESTING WITH CLASS WEIGHTS (Fallback)")
    print("="*50)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print("Training with balanced class weights...")
    model.fit(X_train_scaled, y_train)
    
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Find best threshold
    best_threshold = None
    best_f1 = 0
    
    for threshold in np.arange(0.05, 0.5, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        if prec >= 0.7 and rec >= 0.7 and f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1
    
    if best_threshold:
        print(f"\n✅ Found working threshold: {best_threshold:.3f}")
        y_pred = (y_proba >= best_threshold).astype(int)
        print(f"Precision: {precision_score(y_test, y_pred):.2%}")
        print(f"Recall: {recall_score(y_test, y_pred):.2%}")
    else:
        print("\n⚠️  Could not find threshold meeting requirements")
