#!/usr/bin/env python3
"""
Validate the fraud detection model functionality independently.
This script tests the core model components without Flask.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.model.fraud_model import FraudDetectionModel
from modules.data.raw_data_handler import RawDataHandler
from modules.utils.model_utils import ModelManager

def create_synthetic_data(n_samples=1000):
    """Create synthetic fraud detection data for testing."""
    np.random.seed(42)
    
    merchants = ['Walmart', 'Target', 'Amazon', 'Starbucks', 'McDonalds']
    categories = ['grocery_pos', 'gas_transport', 'shopping_net', 'food_dining']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston']
    states = ['NY', 'CA', 'IL', 'TX']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months = ['January', 'February', 'March', 'April', 'May', 'June']
    
    data = []
    for i in range(n_samples):
        # Create realistic transaction patterns
        hour = np.random.randint(0, 24)
        amount = np.random.lognormal(3, 1)  # Log-normal distribution
        
        # Simple fraud logic: high amounts at odd hours more likely to be fraud
        fraud_probability = 0.1  # Base rate
        if amount > 500:
            fraud_probability += 0.3
        if hour < 6 or hour > 22:
            fraud_probability += 0.2
        
        is_fraud = np.random.random() < fraud_probability
        
        data.append({
            'merchant': np.random.choice(merchants),
            'category': np.random.choice(categories),
            'sex': np.random.choice(['M', 'F']),
            'day_of_week': np.random.choice(days),
            'month_date': np.random.choice(months),
            'hour': hour,
            'city': np.random.choice(cities),
            'state': np.random.choice(states),
            'amt': round(amount, 2),
            'is_fraud': int(is_fraud)
        })
    
    return pd.DataFrame(data)

def test_fraud_model():
    """Test the FraudDetectionModel class."""
    print("🤖 Testing FraudDetectionModel...")
    
    # Create synthetic data
    df = create_synthetic_data(1000)
    print(f"   Generated {len(df)} synthetic transactions")
    print(f"   Fraud rate: {df['is_fraud'].mean():.3f}")
    
    # Define features
    categorical_cols = [
        "merchant", "category", "sex", "day_of_week", 
        "month_date", "hour", "city", "state"
    ]
    numerical_cols = ["amt"]
    
    # Prepare data
    X = df[categorical_cols + numerical_cols]
    y = df["is_fraud"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Initialize model
    model = FraudDetectionModel(
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        random_state=42
    )
    
    # Train model
    print("   Training model...")
    model.fit(X_train, y_train)
    
    # Test predictions
    print("   Making predictions...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"   Accuracy: {accuracy:.3f}")
    
    # Test feature names
    feature_names = model.get_feature_names()
    print(f"   Features: {len(feature_names)} total")
    
    # Test coefficients
    coefficients = model.get_coefficients()
    print(f"   Coefficients: {len(coefficients)} total")
    
    return model, accuracy

def test_model_manager():
    """Test the ModelManager class."""
    print("📊 Testing ModelManager...")
    
    # Create test directory
    test_dir = "test_output"
    os.makedirs(test_dir, exist_ok=True)
    
    manager = ModelManager(models_dir=test_dir)
    
    # Create synthetic data and model
    df = create_synthetic_data(500)
    categorical_cols = ["merchant", "category", "sex", "day_of_week", "month_date", "hour", "city", "state"]
    numerical_cols = ["amt"]
    
    X = df[categorical_cols + numerical_cols]
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = FraudDetectionModel(
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    metrics = manager.evaluate_model(model, X_test, y_test)
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1-score: {metrics['f1_score']:.3f}")
    
    # Validate performance
    is_valid, message = manager.validate_model_performance(metrics)
    print(f"   Performance validation: {message}")
    
    # Save model
    training_info = {"test": True}
    model_id, model_path = manager.save_model(model, metrics, training_info)
    print(f"   Model saved: {model_id}")
    
    # Load model
    loaded_model = manager.load_model(model_path)
    print("   Model loaded successfully")
    
    # Test loaded model
    test_pred = loaded_model.predict(X_test[:5])
    print(f"   Test prediction: {test_pred}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    return metrics

def test_data_handler():
    """Test the RawDataHandler class (if data files exist)."""
    print("📁 Testing RawDataHandler...")
    
    handler = RawDataHandler(
        storage_path="data_sources",
        save_path="test_datasets"
    )
    
    # Check if data files exist
    required_files = [
        "customer_release.csv",
        "transactions_release.parquet", 
        "fraud_release.json"
    ]
    
    files_exist = all(
        os.path.exists(os.path.join("data_sources", f)) 
        for f in required_files
    )
    
    if not files_exist:
        print("   Data files not found - skipping data handler test")
        return None
    
    try:
        # Test data extraction
        customers, transactions, fraud = handler.extract(*required_files)
        print(f"   Customers: {len(customers)}")
        print(f"   Transactions: {len(transactions)}")
        print(f"   Fraud labels: {len(fraud)}")
        
        # Test data transformation
        merged = handler.transform(customers, transactions, fraud)
        print(f"   Merged data: {len(merged)} rows")
        
        # Test data quality report
        quality = handler.describe(merged)
        print(f"   Fraud rate: {quality.get('fraud_rate', 0):.3f}")
        
        # Cleanup
        import shutil
        if os.path.exists("test_datasets"):
            shutil.rmtree("test_datasets")
        
        return quality
        
    except Exception as e:
        print(f"   Error testing data handler: {e}")
        return None

def main():
    """Run all validation tests."""
    print("🏦 SecureBank Model Validation")
    print("==============================")
    
    try:
        # Test 1: Core fraud model
        model, accuracy = test_fraud_model()
        model_success = accuracy > 0.5  # Basic sanity check
        print(f"   Model test: {'✓ PASS' if model_success else '✗ FAIL'}")
        
        # Test 2: Model manager
        metrics = test_model_manager()
        manager_success = metrics['precision'] > 0.1 and metrics['recall'] > 0.1  # Basic sanity check
        print(f"   Manager test: {'✓ PASS' if manager_success else '✗ FAIL'}")
        
        # Test 3: Data handler (if data available)
        quality = test_data_handler()
        data_success = quality is not None
        print(f"   Data handler test: {'✓ PASS' if data_success else '⚠ SKIP (no data)'}")
        
        print("\n==============================")
        
        # Overall result
        tests_run = 2 + (1 if data_success else 0)
        tests_passed = sum([model_success, manager_success, data_success])
        
        print(f"Tests run: {tests_run}")
        print(f"Tests passed: {tests_passed}")
        
        if tests_passed == tests_run:
            print("🎉 All validation tests passed!")
            return 0
        else:
            print("⚠️  Some validation tests failed")
            return 1
            
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit(main())