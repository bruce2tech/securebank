# securebank/modules/utils/advanced_logging.py
"""
Advanced logging utilities for SecureBank system monitoring and analytics.
Extends the basic LoggingManager with analysis, rotation, and monitoring capabilities.
"""

import json
import os
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from collections import defaultdict, Counter
import threading
import time


class LogAnalyzer:
    """
    Analyzes SecureBank log files to provide insights and monitoring data.
    """
    
    def __init__(self, logs_dir: str = "logs"):
        """
        Initialize the log analyzer.
        
        Parameters
        ----------
        logs_dir : str
            Directory containing log files.
        """
        self.logs_dir = logs_dir
    
    def load_logs(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Load log entries from the specified time period.
        
        Parameters
        ----------
        hours_back : int
            Number of hours back to load logs from.
            
        Returns
        -------
        list
            List of log entries as dictionaries.
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        logs = []
        
        # Get all log files
        log_files = glob.glob(os.path.join(self.logs_dir, "log_*.json"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_entry = json.load(f)
                
                # Parse timestamp
                if 'system_timestamp' in log_entry:
                    log_time = datetime.fromisoformat(log_entry['system_timestamp'])
                    if log_time >= cutoff_time:
                        log_entry['_log_file'] = log_file
                        log_entry['_parsed_timestamp'] = log_time
                        logs.append(log_entry)
                        
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Warning: Could not parse log file {log_file}: {e}")
                continue
        
        # Sort by timestamp
        logs.sort(key=lambda x: x.get('_parsed_timestamp', datetime.min))
        return logs
    
    def get_system_metrics(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Generate system performance metrics from logs.
        
        Parameters
        ----------
        hours_back : int
            Hours back to analyze.
            
        Returns
        -------
        dict
            System metrics and statistics.
        """
        logs = self.load_logs(hours_back)
        
        # Initialize counters
        event_counts = Counter()
        endpoint_counts = Counter()
        response_times = []
        error_count = 0
        fraud_predictions = {'fraudulent': 0, 'legitimate': 0}
        
        # Process logs
        for log in logs:
            event_type = log.get('event_type', 'unknown')
            event_counts[event_type] += 1
            
            # Endpoint tracking
            endpoint = log.get('endpoint', 'unknown')
            endpoint_counts[endpoint] += 1
            
            # Performance tracking
            if 'processing_time_seconds' in log:
                response_times.append(log['processing_time_seconds'])
            
            # Error tracking
            if event_type == 'system_error':
                error_count += 1
            
            # Prediction tracking
            if event_type == 'prediction_request':
                prediction = log.get('prediction', 'unknown')
                if prediction in fraud_predictions:
                    fraud_predictions[prediction] += 1
        
        # Calculate statistics
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        
        return {
            "analysis_period_hours": hours_back,
            "total_events": len(logs),
            "event_breakdown": dict(event_counts),
            "endpoint_usage": dict(endpoint_counts),
            "performance_metrics": {
                "avg_response_time_seconds": round(avg_response_time, 3),
                "max_response_time_seconds": round(max_response_time, 3),
                "min_response_time_seconds": round(min_response_time, 3),
                "total_requests_with_timing": len(response_times)
            },
            "error_metrics": {
                "total_errors": error_count,
                "error_rate": round(error_count / len(logs) * 100, 2) if logs else 0
            },
            "fraud_detection_metrics": {
                "predictions": dict(fraud_predictions),
                "fraud_rate": round(
                    fraud_predictions['fraudulent'] / sum(fraud_predictions.values()) * 100, 2
                ) if sum(fraud_predictions.values()) > 0 else 0
            },
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def get_model_performance_history(self) -> List[Dict[str, Any]]:
        """
        Extract model training performance history from logs.
        
        Returns
        -------
        list
            List of model training sessions with performance metrics.
        """
        logs = self.load_logs(hours_back=24*7)  # Last week
        training_sessions = []
        
        for log in logs:
            if log.get('event_type') == 'model_training':
                training_metrics = log.get('training_metrics', {})
                performance = training_metrics.get('performance_metrics', {})
                
                if performance:
                    session = {
                        "timestamp": log.get('system_timestamp'),
                        "model_id": training_metrics.get('model_info', {}).get('model_id'),
                        "precision": performance.get('precision'),
                        "recall": performance.get('recall'),
                        "f1_score": performance.get('f1_score'),
                        "accuracy": performance.get('accuracy'),
                        "training_time": training_metrics.get('training_info', {}).get('training_duration_seconds')
                    }
                    training_sessions.append(session)
        
        return training_sessions
    
    def generate_daily_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive daily activity report.
        
        Parameters
        ----------
        date : datetime, optional
            Date to generate report for. Defaults to today.
            
        Returns
        -------
        dict
            Daily activity report.
        """
        if date is None:
            date = datetime.now()
        
        # Load logs for the specified day
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        all_logs = self.load_logs(hours_back=24*7)  # Load week to find the day
        day_logs = [
            log for log in all_logs 
            if start_time <= log.get('_parsed_timestamp', datetime.min) < end_time
        ]
        
        # Generate hourly breakdown
        hourly_activity = defaultdict(int)
        for log in day_logs:
            hour = log.get('_parsed_timestamp', datetime.min).hour
            hourly_activity[hour] += 1
        
        # Get system metrics for the day
        system_metrics = self.get_system_metrics(hours_back=24)
        
        return {
            "report_date": date.strftime("%Y-%m-%d"),
            "total_activity": len(day_logs),
            "hourly_breakdown": dict(hourly_activity),
            "peak_hour": max(hourly_activity.items(), key=lambda x: x[1])[0] if hourly_activity else None,
            "system_metrics": system_metrics,
            "model_training_sessions": len([
                log for log in day_logs 
                if log.get('event_type') == 'model_training'
            ]),
            "dataset_generations": len([
                log for log in day_logs 
                if log.get('event_type') == 'dataset_creation'
            ]),
            "generated_at": datetime.now().isoformat()
        }


class LogRotationManager:
    """
    Manages log file rotation and cleanup to prevent disk space issues.
    """
    
    def __init__(self, logs_dir: str = "logs", max_age_days: int = 30, max_files: int = 1000):
        """
        Initialize log rotation manager.
        
        Parameters
        ----------
        logs_dir : str
            Directory containing log files.
        max_age_days : int
            Maximum age of log files in days.
        max_files : int
            Maximum number of log files to keep.
        """
        self.logs_dir = logs_dir
        self.max_age_days = max_age_days
        self.max_files = max_files
    
    def cleanup_old_logs(self) -> Dict[str, int]:
        """
        Remove old log files based on age and count limits.
        
        Returns
        -------
        dict
            Cleanup statistics.
        """
        cutoff_time = datetime.now() - timedelta(days=self.max_age_days)
        log_files = glob.glob(os.path.join(self.logs_dir, "log_*.json"))
        
        # Sort by modification time (newest first)
        log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        removed_by_age = 0
        removed_by_count = 0
        
        # Remove files older than max_age_days
        for log_file in log_files[:]:
            file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
            if file_time < cutoff_time:
                try:
                    os.remove(log_file)
                    log_files.remove(log_file)
                    removed_by_age += 1
                except OSError:
                    continue
        
        # Remove excess files if still over limit
        if len(log_files) > self.max_files:
            files_to_remove = log_files[self.max_files:]
            for log_file in files_to_remove:
                try:
                    os.remove(log_file)
                    removed_by_count += 1
                except OSError:
                    continue
        
        return {
            "removed_by_age": removed_by_age,
            "removed_by_count": removed_by_count,
            "remaining_files": len(log_files) - removed_by_count,
            "cleanup_timestamp": datetime.now().isoformat()
        }
    
    def get_log_directory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the log directory.
        
        Returns
        -------
        dict
            Directory statistics.
        """
        log_files = glob.glob(os.path.join(self.logs_dir, "log_*.json"))
        
        if not log_files:
            return {
                "total_files": 0,
                "total_size_mb": 0,
                "oldest_file": None,
                "newest_file": None
            }
        
        # Calculate total size
        total_size = sum(os.path.getsize(f) for f in log_files)
        
        # Find oldest and newest files
        oldest_file = min(log_files, key=lambda x: os.path.getmtime(x))
        newest_file = max(log_files, key=lambda x: os.path.getmtime(x))
        
        return {
            "total_files": len(log_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest_file": {
                "path": oldest_file,
                "age_days": (datetime.now() - datetime.fromtimestamp(
                    os.path.getmtime(oldest_file)
                )).days
            },
            "newest_file": {
                "path": newest_file,
                "age_hours": (datetime.now() - datetime.fromtimestamp(
                    os.path.getmtime(newest_file)
                )).total_seconds() / 3600
            }
        }


class RealTimeLogMonitor:
    """
    Real-time log monitoring for detecting issues and anomalies.
    """
    
    def __init__(self, logs_dir: str = "logs"):
        """
        Initialize real-time monitor.
        
        Parameters
        ----------
        logs_dir : str
            Directory to monitor.
        """
        self.logs_dir = logs_dir
        self.monitoring = False
        self.monitor_thread = None
        self.alerts = []
        self.thresholds = {
            'error_rate_per_hour': 10,
            'avg_response_time_seconds': 5.0,
            'fraud_rate_threshold': 25.0
        }
    
    def start_monitoring(self, check_interval: int = 60):
        """
        Start real-time monitoring.
        
        Parameters
        ----------
        check_interval : int
            Seconds between monitoring checks.
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(check_interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self, check_interval: int):
        """Internal monitoring loop."""
        analyzer = LogAnalyzer(self.logs_dir)
        
        while self.monitoring:
            try:
                # Get recent metrics
                metrics = analyzer.get_system_metrics(hours_back=1)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(check_interval)
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts."""
        current_time = datetime.now()
        
        # Check error rate
        error_rate = metrics['error_metrics']['error_rate']
        if error_rate > self.thresholds['error_rate_per_hour']:
            self.alerts.append({
                "timestamp": current_time.isoformat(),
                "type": "high_error_rate",
                "severity": "warning",
                "message": f"Error rate {error_rate}% exceeds threshold {self.thresholds['error_rate_per_hour']}%",
                "metrics": metrics['error_metrics']
            })
        
        # Check response time
        avg_response_time = metrics['performance_metrics']['avg_response_time_seconds']
        if avg_response_time > self.thresholds['avg_response_time_seconds']:
            self.alerts.append({
                "timestamp": current_time.isoformat(),
                "type": "slow_response_time",
                "severity": "warning",
                "message": f"Average response time {avg_response_time}s exceeds threshold {self.thresholds['avg_response_time_seconds']}s",
                "metrics": metrics['performance_metrics']
            })
        
        # Check fraud rate
        fraud_rate = metrics['fraud_detection_metrics']['fraud_rate']
        if fraud_rate > self.thresholds['fraud_rate_threshold']:
            self.alerts.append({
                "timestamp": current_time.isoformat(),
                "type": "high_fraud_rate",
                "severity": "info",
                "message": f"Fraud detection rate {fraud_rate}% exceeds threshold {self.thresholds['fraud_rate_threshold']}%",
                "metrics": metrics['fraud_detection_metrics']
            })
        
        # Keep only recent alerts (last 24 hours)
        cutoff = current_time - timedelta(hours=24)
        self.alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) >= cutoff
        ]
    
    def get_recent_alerts(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Parameters
        ----------
        hours_back : int
            Hours back to retrieve alerts.
            
        Returns
        -------
        list
            List of recent alerts.
        """
        cutoff = datetime.now() - timedelta(hours=hours_back)
        return [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) >= cutoff
        ]