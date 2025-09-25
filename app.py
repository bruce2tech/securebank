# securebank/app.py
"""
SecureBank Fraud Detection System API
Fixed version with consistent variable naming and LightGBM support
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import time
import logging
import traceback
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from modules.utils.logging_utils import LoggingManager
from modules.utils.advanced_logging import LogAnalyzer, RealTimeLogMonitor
from modules.utils.data_drift_detector import SecureBankDriftMonitor
import glob

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for model
model = None
model_type = None
feature_columns = []

# Directories
MODEL_DIR = "storage/models"
DATASET_DIR = "storage/datasets"
LOG_DIR = "logs"
OUTPUT_DIR = "output"


# Create necessary directories
for directory in [MODEL_DIR, DATASET_DIR, LOG_DIR, OUTPUT_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

def create_synthetic_data(n_samples=10000):
    """Create synthetic dataset for testing"""
    np.random.seed(42)
    
    data = {
        'amt': np.random.lognormal(3.5, 1.5, n_samples),
        'unix_time': np.random.randint(1577836800, 1609459200, n_samples),
        'merch_lat': np.random.uniform(25, 50, n_samples),
        'merch_long': np.random.uniform(-125, -65, n_samples),
        'cc_num': np.random.randint(1000000000000000, 9999999999999999, n_samples)
    }
    
    # Add more features if needed
    for i in range(5, 62):  # Create 62 features total
        data[f'feature_{i}'] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)

engineered_path = Path(DATASET_DIR) / "dataset_engineered_raw.csv"
if not engineered_path.exists():
    logger.info("Creating engineered dataset from raw data...")
    try:
        from engineer_from_raw import engineer_features_from_raw
        df = engineer_features_from_raw()
        df.to_csv(engineered_path, index=False)
        logger.info(f"Created engineered dataset with {len(df.columns)} features, fraud rate: {df['is_fraud'].mean():.2%}")
    except Exception as e:
        logger.error(f"Could not create engineered dataset: {e}")
        # Fall back to synthetic data
        logger.info("Falling back to synthetic data generation...")
        df = create_synthetic_data(1000000)  # Create large synthetic dataset
        df.to_csv(engineered_path, index=False)

def load_default_model():
    """Load the default model at startup"""
    global model, model_type, feature_columns
    
    model_loaded = False
    
    # Check multiple possible locations for model
    possible_paths = [
        Path("output/best_lgb_model.pkl"),
        Path("output/fraud_model_lightgbm_adasyn.pkl"),
        Path("storage/models/fraud_model_lightgbm_adasyn.pkl"),
        Path("storage/models/best_lgb_model.pkl")
    ]
    
    for model_path in possible_paths:
        if model_path.exists():
            try:
                logger.info(f"Loading model from {model_path}...")
                
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                if isinstance(model_data, dict):
                    model = model_data.get('model')
                    feature_columns = model_data.get('feature_names', model_data.get('feature_cols', []))
                    model_type = model_data.get('model_type', 'unknown')
                    performance = model_data.get('performance', {})
                    
                    if model:
                        logger.info(f"Loaded {model_type} model with {len(feature_columns)} features")
                        if performance:
                            logger.info(f"Model performance - Precision: {performance.get('precision', 0):.2%}, Recall: {performance.get('recall', 0):.2%}")
                        model_loaded = True
                        break
                else:
                    # Direct model object
                    model = model_data
                    feature_columns = []
                    model_type = "unknown"
                    logger.info("Loaded model (direct format)")
                    model_loaded = True
                    break
                    
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {str(e)}")
                continue
    
    if not model_loaded:
        logger.warning("No model could be loaded at startup")
        logger.warning("Train a model using POST /train_model")
        model = None
        model_type = None
        feature_columns = []
    
    return model_loaded

# Load model at module level (works in Docker)
logger.info("Starting SecureBank Fraud Detection System...")
load_default_model()

logger.info("Initializing monitoring components...")
logger_manager = LoggingManager(logs_dir=LOG_DIR)

# Debug code to testlogger_manager initialization:
print(f"LoggingManager initialized: {logger_manager is not None}")
print(f"LoggingManager logs_dir: {logger_manager.logs_dir}")
print(f"Logs directory exists: {os.path.exists(logger_manager.logs_dir)}")

# Test write directly
test_path = logger_manager.log_prediction_request(
    request_data={"debug": "test"},
    prediction="test",
    probability=0.5,
    processing_time=0.001
)
print(f"Test log created at: {test_path}")
print(f"Test log exists: {os.path.exists(test_path)}")

log_analyzer = LogAnalyzer(logs_dir=LOG_DIR)
drift_monitor = SecureBankDriftMonitor(data_path=DATASET_DIR)
real_time_monitor = RealTimeLogMonitor(logs_dir=LOG_DIR)

# Initialize drift monitoring if baseline exists
baseline_path = Path(DATASET_DIR) / "dataset_engineered_raw.csv"
if baseline_path.exists():
    try:
        drift_monitor.initialize_monitoring(str(baseline_path))
        logger.info("Drift monitoring initialized")
    except Exception as e:
        logger.warning(f"Could not initialize drift monitoring: {e}")

# Start real-time monitoring
real_time_monitor.start_monitoring(check_interval=60)
logger.info("Real-time monitoring started")




def log_request(endpoint: str, request_data: dict, response_data: dict, error: str = None):
    """Log API requests and responses"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "request": request_data,
        "response": response_data,
        "error": error
    }
    
    log_file = Path(LOG_DIR) / f"log_{timestamp}.json"
    with open(log_file, 'w') as f:
        json.dump(log_entry, f, indent=2, default=str)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    if model is not None:
        status["model_info"] = {
            "type": model_type,
            "feature_count": len(feature_columns)
        }
    
    return jsonify(status), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Predict fraud for a transaction"""
    start_time = time.time()
    try:
        # Get transaction data
        transaction_data = request.get_json()
        
        if not transaction_data:
            error_msg = "No JSON data provided"
            # For errors, use logger_manager.log_system_error
            logger_manager.log_system_error({
                "endpoint": "/predict",
                "error": error_msg,
                "request_data": {}
            })
            return jsonify({"error": error_msg}), 400
        
        # Check if model is loaded
        if model is None:
            error_msg = "No model available. Please train a model first."
            logger_manager.log_system_error({
                "endpoint": "/predict",
                "error": error_msg,
                "request_data": transaction_data
            })
            return jsonify({"error": error_msg}), 503
        
        # Create DataFrame from transaction
        trans_df = pd.DataFrame([transaction_data])
        
        # Handle different model types
        if model_type == "lightgbm":
            # For LightGBM Booster
            # Ensure we have all required features
            for feat in feature_columns:
                if feat not in trans_df.columns:
                    # Add missing features with default values
                    if 'encoded' in feat or 'category' in feat:
                        trans_df[feat] = 0
                    elif 'lat' in feat or 'long' in feat:
                        trans_df[feat] = 0.0
                    else:
                        trans_df[feat] = 0.0
            
            # Select only the features the model expects, in the right order
            trans_df = trans_df[feature_columns]
            
            # Make prediction
            import lightgbm as lgb
            if isinstance(model, lgb.Booster):
                fraud_probability = model.predict(trans_df.values)[0]
            else:
                # Fallback for other model types
                fraud_probability = 0.5
                
        elif hasattr(model, 'predict_proba'):
            # For scikit-learn models
            # Prepare features
            for feat in feature_columns:
                if feat not in trans_df.columns:
                    trans_df[feat] = 0
            trans_df = trans_df[feature_columns]
            
            # Make prediction
            fraud_probability = model.predict_proba(trans_df.values)[0][1]
            
        elif hasattr(model, 'predict'):
            # For models with only predict method
            for feat in feature_columns:
                if feat not in trans_df.columns:
                    trans_df[feat] = 0
            trans_df = trans_df[feature_columns]
            
            prediction = model.predict(trans_df.values)[0]
            fraud_probability = float(prediction)
        else:
            # Unknown model type
            fraud_probability = 0.5
        
        # Determine result (threshold = 0.5)
        is_fraud = fraud_probability > 0.5
        
        response = {
            "predict": "fraudulent" if is_fraud else "legitimate",
            "confidence": float(fraud_probability if is_fraud else 1 - fraud_probability),
            "fraud_probability": float(fraud_probability),
            "model_type": model_type,
            "features_used": len(feature_columns)
        }
        
        processing_time = time.time() - start_time  # Add start_time = time.time() at beginning of function
        logger_manager.log_prediction_request(
            request_data=transaction_data,
            prediction=response["predict"],
            probability=response["fraud_probability"],
            processing_time=processing_time
            )
        return jsonify(response), 200
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Prediction error: {traceback.format_exc()}")
        # For exceptions, use logger_manager.log_system_error
        logger_manager.log_system_error({
            "endpoint": "/predict",
            "error": error_msg,
            "traceback": traceback.format_exc(),
            "request_data": transaction_data if 'transaction_data' in locals() else {}
        })
        return jsonify({"error": error_msg}), 500

@app.route('/create_dataset', methods=['POST'])
def create_dataset():
    """Create a new dataset for training"""
    try:
        params = request.get_json() or {}
        
        # Try to load raw data if available
        data_sources_dir = Path("data_sources")
        if data_sources_dir.exists():
            # Load any available data files
            transactions = None
            customers = None
            
            # Look for transaction data
            for file in data_sources_dir.glob("*transaction*"):
                if file.suffix == '.parquet':
                    transactions = pd.read_parquet(file)
                elif file.suffix == '.csv':
                    transactions = pd.read_csv(file)
                break
            
            # Look for customer data
            for file in data_sources_dir.glob("*customer*"):
                if file.suffix == '.csv':
                    customers = pd.read_csv(file)
                break
            
            if transactions is not None:
                df = transactions
                if customers is not None and 'cc_num' in transactions.columns and 'cc_num' in customers.columns:
                    # Merge if possible
                    df = pd.merge(transactions, customers, on='cc_num', how='left')
            else:
                # Create synthetic data
                df = create_synthetic_data()
        else:
            # Create synthetic data
            df = create_synthetic_data()
        
        # Add fraud labels if not present
        if 'is_fraud' not in df.columns:
            np.random.seed(42)
            df['is_fraud'] = np.random.choice([0, 1], size=len(df), p=[0.99, 0.01])
        
        # Save dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_path = Path(DATASET_DIR) / f"dataset_train_{timestamp}.csv"
        df.to_csv(dataset_path, index=False)
        
        response = {
            "status": "success",
            "dataset_path": str(dataset_path),
            "rows": len(df),
            "columns": len(df.columns),
            "fraud_ratio": float(df['is_fraud'].mean()) if 'is_fraud' in df.columns else 0.01,
            "timestamp": timestamp
        }
        
        log_request("/create_dataset", params, response)
        return jsonify(response), 200
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Dataset creation error: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

@app.route('/train_model', methods=['POST'])
def train_model_endpoint():
    """Train a new fraud detection model using the proven LightGBM configuration"""
    start_time = time.time()
    
    global model, model_type, feature_columns
    
    try:
        logger.info("Starting model training with proven LightGBM configuration...")
        
        # Load the engineered dataset (prefer the 77-feature one)
        dataset_path = Path("storage/datasets/dataset_engineered_raw.csv")
        if not dataset_path.exists():
            # Fall back to latest dataset
            datasets = sorted(Path("storage/datasets").glob("dataset_*.csv"))
            if not datasets:
                return jsonify({"error": "No dataset available"}), 404
            dataset_path = datasets[-1]
        
        logger.info(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # Prepare features - exclude non-feature columns
        exclude_cols = [
            'trans_num', 'cc_num', 'trans_date_trans_time', 'is_fraud',
            'merchant', 'category', 'gender', 'city', 'state', 'job',
            'first', 'last', 'street', 'zip', 'dob', 'date',
            'prev_lat', 'prev_long'
        ]
        
        if 'is_fraud' not in df.columns:
            return jsonify({'error': 'No fraud labels in dataset'}), 400
        
        # Select numeric features only
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols and col in df.columns:
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    feature_cols.append(col)
        
        X = df[feature_cols].fillna(0)
        y = df['is_fraud']
        
        logger.info(f"Dataset shape: {X.shape}, Fraud rate: {y.mean():.2%}")
        logger.info(f"Using {len(feature_cols)} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply ADASYN with 3% sampling (the PROVEN configuration)
        try:
            from imblearn.over_sampling import ADASYN
            
            adasyn = ADASYN(
                sampling_strategy=0.03,  # 3% as proven in test
                random_state=42,
                n_neighbors=5
            )
            X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
            logger.info(f"ADASYN applied: {len(X_resampled)} samples, fraud rate: {y_resampled.mean():.2%}")
            adasyn_used = True
        except Exception as e:
            logger.warning(f"ADASYN not available or failed: {e}")
            X_resampled, y_resampled = X_train, y_train
            adasyn_used = False
        
        # Train LightGBM with the EXACT Aggressive configuration that worked
        try:
            import lightgbm as lgb
            logger.info("Training LightGBM with proven Aggressive configuration...")
            
            # EXACT parameters from successful test
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'dart',  # DART boosting as proven
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
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_resampled, label=y_resampled)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            # Train with same settings as script
            trained_model = lgb.train(
                params,
                train_data,
                num_boost_round=300,  # Same as script
                valid_sets=[valid_data],
                callbacks=[
                    lgb.early_stopping(50),  # Same as script
                    lgb.log_evaluation(0)
                ]
            )
            
            model_type_used = "lightgbm"
            
            # Get predictions with threshold 0.5 (proven threshold)
            y_pred_proba = trained_model.predict(X_test, num_iteration=trained_model.best_iteration)
            threshold = 0.5  # proven threshold
            y_pred = (y_pred_proba >= threshold).astype(int)
            
        except ImportError:
            logger.warning("LightGBM not available, falling back to RandomForest")
            from sklearn.ensemble import RandomForestClassifier
            
            trained_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            trained_model.fit(X_resampled, y_resampled)
            model_type_used = "random_forest"
            y_pred = trained_model.predict(X_test)
            threshold = 0.5
        
        # Calculate metrics - ensure they're proper floats
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        
        logger.info(f"Model Performance - Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")
        
        # Update global variables
        model = trained_model
        model_type = model_type_used
        feature_columns = feature_cols
        
        # Save model with all metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = Path(MODEL_DIR) / f"fraud_model_{timestamp}.pkl"
        
        model_data = {
            'model': trained_model,
            'feature_names': feature_columns,
            'feature_count': len(feature_columns),
            'model_type': model_type_used,
            'threshold': threshold,
            'performance': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'params': params if model_type_used == "lightgbm" else {},
            'adasyn_used': adasyn_used,
            'sampling_strategy': 0.03 if adasyn_used else None,
            'trained_at': timestamp
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f, protocol=2)
        
        # Also save as best model
        best_model_path = Path(MODEL_DIR) / "best_lgb_model.pkl"
        with open(best_model_path, 'wb') as f:
            pickle.dump(model_data, f, protocol=2)
        
        # Save the actual LightGBM model in text format if applicable
        if model_type_used == "lightgbm":
            text_model_path = Path(MODEL_DIR) / "lightgbm_model.txt"
            trained_model.save_model(str(text_model_path))
            
            # Save like script does
            script_model_path = Path(f"storage/models/fraud_model_lightgbm_adasyn_{timestamp}.pkl.txt")
            script_model_path.parent.mkdir(parents=True, exist_ok=True)
            trained_model.save_model(str(script_model_path))
            
            # Save metadata like script
            import joblib
            joblib.dump({
                'threshold': threshold,
                'feature_cols': feature_columns,
                'params': params
            }, Path(f"storage/models/fraud_model_lightgbm_adasyn_{timestamp}.pkl"))
        
        # Log training event
        log_data = {
            'timestamp': timestamp,
            'model_type': model_type_used,
            'configuration': 'Aggressive' if model_type_used == "lightgbm" else 'Default',
            'adasyn_sampling': 0.03 if adasyn_used else None,
            'threshold': threshold,
            'performance': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'features_count': len(feature_columns),
            'training_samples': len(X_resampled),
            'test_samples': len(X_test),
            'processing_time': time.time() - start_time
        }
        
        log_path = Path(LOG_DIR) / f"log_{timestamp}.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Return response
        meets_requirements = precision >= 0.7 and recall >= 0.7
        
        drift_analysis = None

        # Uncomment to conduct drift monitoring on new data.
        # if drift_monitor.detector.baseline_stats:
        #     try:
        #         dataset_path = Path("storage/datasets/dataset_engineered_raw.csv")
        #         if dataset_path.exists():
        #             drift_report = drift_monitor.check_new_data_drift(str(dataset_path))
        #             drift_analysis = {
        #                 "drift_detected": drift_report.features_with_drift > 0,
        #                 "drift_percentage": drift_report.drift_percentage,
        #                 "severity": drift_report.overall_severity,
        #                 "recommendation": drift_report.recommendations[0] if drift_report.recommendations else "No concerns"
        #             }
        #     except Exception as e:
        #         logger.warning(f"Could not perform drift analysis: {e}")

        # Log training with drift analysis
        logger_manager.log_model_training({
            "model_id": f"model_{timestamp}",
            "timestamp": timestamp,
            "performance": {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            "drift_analysis": drift_analysis
        })

        return jsonify({
            'status': 'success',
            'model_path': str(model_path),
            'model_type': model_type_used,
            'configuration': 'LightGBM-Aggressive' if model_type_used == "lightgbm" else model_type_used,
            'threshold': threshold,
            'adasyn_used': adasyn_used,
            'performance': {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4)
            },
            'meets_requirements': meets_requirements,
            'message': f'Model {"MEETS" if meets_requirements else "does not meet"} 70/70 requirements',
            'features_used': len(feature_columns),
            'training_samples': len(X_resampled),
            'test_samples': len(X_test),
            'processing_time_seconds': round(time.time() - start_time, 2)
        }), 200
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Training failed', 'message': str(e)}), 500

@app.route('/system_metrics', methods=['GET'])
def get_system_metrics():
    """Get system performance metrics from logs"""
    try:
        hours_back = request.args.get('hours', 24, type=int)
        metrics = log_analyzer.get_system_metrics(hours_back=hours_back)
        return jsonify(metrics), 200
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/drift_check', methods=['POST'])
def check_drift():
    """Check for data drift in new dataset"""
    try:
        dataset_files = list(Path(DATASET_DIR).glob("dataset_*.csv"))
        if not dataset_files:
            return jsonify({"error": "No datasets found"}), 404
        
        latest_dataset = max(dataset_files, key=lambda p: p.stat().st_mtime)
        
        if not drift_monitor.detector.baseline_stats:
            baseline = min(dataset_files, key=lambda p: p.stat().st_mtime)
            drift_monitor.initialize_monitoring(str(baseline))
        
        report = drift_monitor.check_new_data_drift(str(latest_dataset))
        
        return jsonify({
            "drift_detected": report.features_with_drift > 0,
            "drift_percentage": report.drift_percentage,
            "severity": report.overall_severity,
            "features_with_drift": report.features_with_drift,
            "recommendations": report.recommendations
        }), 200
        
    except Exception as e:
        logger.error(f"Error checking drift: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/monitoring_status', methods=['GET'])
def monitoring_status():
    """Get current monitoring and drift detection status"""
    try:
        return jsonify({
            "drift_monitoring": drift_monitor.get_monitoring_status(),
            "log_metrics": log_analyzer.get_system_metrics(hours_back=1),
            "recent_alerts": real_time_monitor.get_recent_alerts(hours_back=24),
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error getting monitoring status: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/test_logging', methods=['GET'])
def test_logging():
    """Test if logging manager works"""
    import datetime
    
    # Try to log directly
    test_log_path = logger_manager.log_prediction_request(
        request_data={"test": "data"},
        prediction="legitimate",
        probability=0.1,
        processing_time=0.005
    )
    
    return jsonify({
        "logger_exists": logger_manager is not None,
        "log_dir": logger_manager.logs_dir if hasattr(logger_manager, 'logs_dir') else "unknown",
        "test_log_created": test_log_path,
        "current_time": datetime.datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Only runs when executed directly, not in Docker import
    app.run(host='0.0.0.0', port=5000, debug=False)