# securebank/modules/utils/model_utils.py
"""
Model management utilities for SecureBank fraud detection system.
"""

import os
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


class ModelManager:
    """
    Manages model versioning, serialization, and performance evaluation.
    
    Provides functionality for:
    - Model serialization with versioning
    - Performance evaluation and reporting
    - Model loading and validation
    - Model metadata management
    """
    
    def __init__(self, models_dir: str = "output"):
        """
        Initialize the model manager.
        
        Parameters
        ----------
        models_dir : str, default="output"
            Directory path where trained models will be stored.
        """
        self.models_dir = models_dir
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
    
    def _generate_model_id(self, performance_metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Generate a unique model identifier based on timestamp and performance.
        
        Parameters
        ----------
        performance_metrics : dict, optional
            Model performance metrics to include in identifier.
            
        Returns
        -------
        str
            Unique model identifier.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if performance_metrics:
            precision = performance_metrics.get('precision', 0)
            recall = performance_metrics.get('recall', 0)
            return f"fraud_model_{timestamp}_p{precision:.3f}_r{recall:.3f}"
        else:
            return f"fraud_model_{timestamp}"
    
    def evaluate_model(self, model, X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Parameters
        ----------
        model : sklearn model
            Trained model to evaluate.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test labels.
            
        Returns
        -------
        dict
            Dictionary containing performance metrics.
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Fraud probability
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(accuracy),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "total_samples": len(y_test),
            "fraud_rate": float(y_test.mean())
        }
    
    def save_model(self, model, performance_metrics: Dict[str, float], 
                   training_info: Dict[str, Any]) -> Tuple[str, str]:
        """
        Save trained model with versioning and metadata.
        
        Parameters
        ----------
        model : sklearn model
            Trained model to save.
        performance_metrics : dict
            Model performance metrics.
        training_info : dict
            Additional training information.
            
        Returns
        -------
        tuple
            (model_id, model_path) - Model identifier and file path.
        """
        # Generate unique model ID
        model_id = self._generate_model_id(performance_metrics)
        
        # Save model pickle file
        model_filename = f"{model_id}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            "model_id": model_id,
            "created_timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "performance_metrics": performance_metrics,
            "training_info": training_info,
            "model_type": str(type(model)).split("'")[1]  # Extract class name
        }
        
        metadata_filename = f"{model_id}_metadata.json"
        metadata_path = os.path.join(self.models_dir, metadata_filename)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return model_id, model_path
    
    def load_model(self, model_path: str):
        """
        Load a trained model from file.
        
        Parameters
        ----------
        model_path : str
            Path to the model pickle file.
            
        Returns
        -------
        sklearn model
            Loaded trained model.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def get_latest_model_path(self) -> Optional[str]:
        """
        Get the path to the most recently created model.
        
        Returns
        -------
        str or None
            Path to the latest model file, or None if no models exist.
        """
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        
        if not model_files:
            return None
        
        # Sort by filename (which includes timestamp)
        model_files.sort(reverse=True)
        return os.path.join(self.models_dir, model_files[0])
    
    def validate_model_performance(self, performance_metrics: Dict[str, float],
                                 min_precision: float = 0.7,
                                 min_recall: float = 0.7) -> Tuple[bool, str]:
        """
        Validate that model meets minimum performance requirements.
        
        Parameters
        ----------
        performance_metrics : dict
            Model performance metrics.
        min_precision : float, default=0.7
            Minimum required precision.
        min_recall : float, default=0.7
            Minimum required recall.
            
        Returns
        -------
        tuple
            (is_valid, message) - Whether model meets requirements and status message.
        """
        precision = performance_metrics.get('precision', 0)
        recall = performance_metrics.get('recall', 0)
        
        if precision >= min_precision and recall >= min_recall:
            return True, f"Model meets requirements (Precision: {precision:.3f}, Recall: {recall:.3f})"
        else:
            return False, f"Model below requirements (Precision: {precision:.3f}, Recall: {recall:.3f}). Required: P>={min_precision}, R>={min_recall}"