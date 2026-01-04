"""Data handling module."""

try:
    from .raw_data_handler import RawDataHandler
except ImportError:
    # Fallback if the class doesn't exist
    pass

__all__ = ['RawDataHandler']
