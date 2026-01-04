"""Model module for fraud detection."""

# Import only what actually exists in fraud_model.py
try:
    from .fraud_model import FraudDetectionModel, ModelValidator
except ImportError:
    # Fallback if those classes don't exist
    pass

__all__ = ['FraudDetectionModel', 'ModelValidator']
