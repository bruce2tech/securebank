"""Feature engineering module."""

try:
    from .feature_engineer import FeatureEngineer, DataMerger
except ImportError:
    # Fallback if those classes don't exist
    pass

__all__ = ['FeatureEngineer', 'DataMerger']
