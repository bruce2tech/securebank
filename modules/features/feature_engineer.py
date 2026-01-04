# AUTO-GENERATED: Fixed Feature Engineering Module
# This file contains the corrected feature engineering pipeline
# that prevents duplicate columns and ensures consistent features

# securebank/modules/features/feature_engineer.py
"""
Fixed Feature Engineering module for SecureBank fraud detection system.
Ensures consistent feature names between training and prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import json

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Manages feature creation and ensures consistency between training and inference.
    """
    
    def __init__(self):
        self.feature_schema = None
        self.feature_names = None
        self.categorical_features = [
            'merchant', 'category', 'gender', 'city', 'state', 'job', 
            'hour', 'day_of_week'
        ]
        self.numerical_features = ['amt', 'age', 'city_pop', 'hour_sin', 'hour_cos']
        
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Learn the feature schema from training data.
        """
        # Process features to establish schema
        df_processed = self._create_base_features(df.copy())
        
        # Store the exact feature names and types
        self.feature_schema = {
            'columns': list(df_processed.columns),
            'dtypes': df_processed.dtypes.to_dict(),
            'categorical': self.categorical_features,
            'numerical': self.numerical_features
        }
        
        self.feature_names = list(df_processed.columns)
        
        logger.info(f"Feature schema established with {len(self.feature_names)} features")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data ensuring exact feature matching with training schema.
        """
        # Create base features
        df_processed = self._create_base_features(df.copy())
        
        # Ensure all expected features exist
        if self.feature_schema:
            df_processed = self._align_to_schema(df_processed)
        
        return df_processed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the feature schema and transform in one step.
        """
        self.fit(df)
        return self.transform(df)
    
    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all base features without causing duplicates.
        """
        # Ensure we have clean column names first
        df = self._clean_column_names(df)
        
        # Extract temporal features if we have datetime
        if 'trans_date_trans_time' in df.columns:
            df = self._extract_temporal_features(df)
        
        # Create derived features
        df = self._create_derived_features(df)
        
        # Select only the features we want to use
        feature_cols = self._get_feature_columns(df)
        df_features = df[feature_cols].copy()
        
        return df_features
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names to prevent duplicates.
        """
        # Remove any _x or _y suffixes from previous merges
        df.columns = [col.replace('_x', '').replace('_y', '') 
                     if col.endswith(('_x', '_y')) else col 
                     for col in df.columns]
        
        # Handle duplicate columns by keeping first occurrence
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from transaction datetime.
        """
        try:
            # Convert to datetime if it's not already
            if df['trans_date_trans_time'].dtype == 'object':
                df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
            
            # Extract hour and day of week
            df['hour'] = df['trans_date_trans_time'].dt.hour
            df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
            
            # Create cyclical features for hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
        except Exception as e:
            logger.warning(f"Could not extract temporal features: {e}")
            # Create default values
            df['hour'] = 0
            df['day_of_week'] = 0
            df['hour_sin'] = 0
            df['hour_cos'] = 1
            
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features safely.
        """
        # Amount features
        if 'amt' in df.columns:
            # Log transform for amount (handle zeros)
            df['amt_log'] = np.log1p(df['amt'].fillna(0))
        
        # Customer age feature (if birth year exists)
        if 'dob' in df.columns:
            try:
                current_year = datetime.now().year
                birth_year = pd.to_datetime(df['dob']).dt.year
                df['age'] = current_year - birth_year
            except:
                df['age'] = 35  # Default age
        elif 'age' not in df.columns:
            df['age'] = 35
            
        return df
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get the list of feature columns to use.
        """
        # Base features we always want
        base_features = [
            'amt', 'merchant', 'category', 'gender', 
            'city', 'state', 'job', 'age', 'city_pop'
        ]
        
        # Temporal features
        temporal_features = ['hour', 'day_of_week', 'hour_sin', 'hour_cos']
        
        # Derived features
        derived_features = ['amt_log']
        
        # Combine all features
        all_features = base_features + temporal_features + derived_features
        
        # Only include features that exist in the dataframe
        available_features = [f for f in all_features if f in df.columns]
        
        return available_features
    
    def _align_to_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align dataframe to match the training schema exactly.
        """
        if not self.feature_schema:
            return df
            
        expected_cols = self.feature_schema['columns']
        
        # Add missing columns with appropriate defaults
        for col in expected_cols:
            if col not in df.columns:
                # Determine appropriate default based on feature type
                if col in self.categorical_features:
                    df[col] = 'unknown'
                else:
                    df[col] = 0
                    
        # Remove extra columns
        df = df[expected_cols]
        
        # Ensure correct dtypes
        for col, dtype in self.feature_schema['dtypes'].items():
            if col in df.columns:
                try:
                    if 'float' in str(dtype):
                        df[col] = df[col].astype(float)
                    elif 'int' in str(dtype):
                        df[col] = df[col].fillna(0).astype(int)
                    else:
                        df[col] = df[col].astype(str)
                except:
                    pass
                    
        return df
    
    def save_schema(self, filepath: str):
        """
        Save the feature schema to a JSON file.
        """
        if self.feature_schema:
            schema_to_save = {
                'columns': self.feature_schema['columns'],
                'dtypes': {k: str(v) for k, v in self.feature_schema['dtypes'].items()},
                'categorical': self.feature_schema['categorical'],
                'numerical': self.feature_schema['numerical']
            }
            
            with open(filepath, 'w') as f:
                json.dump(schema_to_save, f, indent=2)
                
            logger.info(f"Feature schema saved to {filepath}")
    
    def load_schema(self, filepath: str):
        """
        Load feature schema from a JSON file.
        """
        with open(filepath, 'r') as f:
            schema = json.load(f)
            
        self.feature_schema = schema
        self.feature_names = schema['columns']
        self.categorical_features = schema['categorical']
        self.numerical_features = schema['numerical']
        
        logger.info(f"Feature schema loaded from {filepath}")


class DataMerger:
    """
    Handles merging of different data sources without creating duplicates.
    """
    
    @staticmethod
    def merge_transaction_customer(transactions: pd.DataFrame, 
                                  customers: pd.DataFrame) -> pd.DataFrame:
        """
        Merge transaction and customer data without creating duplicate columns.
        """
        # Identify overlapping columns (excluding join key)
        join_key = 'cc_num'
        overlap_cols = set(transactions.columns) & set(customers.columns)
        overlap_cols.discard(join_key)
        
        if overlap_cols:
            # Rename overlapping columns in customers before merge
            customer_rename = {col: f"cust_{col}" for col in overlap_cols}
            customers = customers.rename(columns=customer_rename)
            logger.info(f"Renamed overlapping columns: {customer_rename}")
        
        # Perform merge
        merged = transactions.merge(
            customers,
            on=join_key,
            how='left'
        )
        
        # Handle any remaining duplicates
        merged = merged.loc[:, ~merged.columns.duplicated()]
        
        return merged
    
    @staticmethod
    def merge_with_labels(data: pd.DataFrame, 
                         labels: pd.DataFrame) -> pd.DataFrame:
        """
        Merge data with fraud labels safely.
        """
        if 'is_fraud' in data.columns:
            # Remove existing fraud column to prevent duplicates
            data = data.drop(columns=['is_fraud'])
            
        # Ensure index-based merge
        if 'index' in labels.columns:
            data = data.merge(
                labels[['index', 'is_fraud']],
                left_index=True,
                right_on='index',
                how='left'
            ).drop(columns=['index'])
        else:
            # Assume labels are aligned by index
            data['is_fraud'] = labels['is_fraud'].values[:len(data)]
            
        return data
