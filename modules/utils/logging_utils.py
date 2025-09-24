# securebank/modules/utils/logging_utils.py
"""
Logging utilities for SecureBank system monitoring and audit trails.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any
import threading


class LoggingManager:
    """
    Manages JSON-based logging for SecureBank system.
    
    Provides thread-safe logging capabilities for:
    - Prediction requests and responses
    - Training activities and metrics
    - Dataset generation statistics
    - System monitoring data
    """
    
    def __init__(self, logs_dir: str = "logs"):
        """
        Initialize the logging manager.
        
        Parameters
        ----------
        logs_dir : str, default="logs"
            Directory path where log files will be stored.
        """
        self.logs_dir = logs_dir
        self.lock = threading.Lock()
        
        # Ensure logs directory exists
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def _get_timestamp(self) -> str:
        """Generate timestamp string for log filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
    
    def _write_log(self, log_data: Dict[str, Any]) -> str:
        """
        Write log data to a JSON file with timestamp.
        
        Parameters
        ----------
        log_data : dict
            Dictionary containing log information.
            
        Returns
        -------
        str
            Path to the created log file.
        """
        timestamp = self._get_timestamp()
        filename = f"log_{timestamp}.json"
        filepath = os.path.join(self.logs_dir, filename)
        
        # Add system timestamp to log data
        log_data["system_timestamp"] = datetime.now().isoformat()
        
        with self.lock:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)
                return filepath
            except Exception as e:
                print(f"Error writing log file {filepath}: {e}")
                return ""
    
    def log_prediction_request(self, request_data: Dict[str, Any], 
                             prediction: str, probability: float,
                             processing_time: float) -> str:
        """
        Log a prediction request and response.
        
        Parameters
        ----------
        request_data : dict
            Input transaction data for prediction.
        prediction : str
            Prediction result ("legitimate" or "fraudulent").
        probability : float
            Fraud probability score.
        processing_time : float
            Time taken to process the request in seconds.
            
        Returns
        -------
        str
            Path to the created log file.
        """
        log_data = {
            "event_type": "prediction_request",
            "request_data": request_data,
            "prediction": prediction,
            "fraud_probability": probability,
            "processing_time_seconds": processing_time,
            "endpoint": "/predict"
        }
        return self._write_log(log_data)
    
    def log_dataset_creation(self, dataset_info: Dict[str, Any]) -> str:
        """
        Log dataset creation activity.
        
        Parameters
        ----------
        dataset_info : dict
            Information about the created dataset.
            
        Returns
        -------
        str
            Path to the created log file.
        """
        log_data = {
            "event_type": "dataset_creation",
            "dataset_info": dataset_info,
            "endpoint": "/create_dataset"
        }
        return self._write_log(log_data)
    
    def log_model_training(self, training_metrics: Dict[str, Any]) -> str:
        """
        Log model training activity and results.
        
        Parameters
        ----------
        training_metrics : dict
            Training metrics and model information.
            
        Returns
        -------
        str
            Path to the created log file.
        """
        log_data = {
            "event_type": "model_training",
            "training_metrics": training_metrics,
            "endpoint": "/train_model"
        }
        return self._write_log(log_data)
    
    def log_system_error(self, error_info: Dict[str, Any]) -> str:
        """
        Log system errors and exceptions.
        
        Parameters
        ----------
        error_info : dict
            Error information and context.
            
        Returns
        -------
        str
            Path to the created log file.
        """
        log_data = {
            "event_type": "system_error",
            "error_info": error_info
        }
        return self._write_log(log_data)