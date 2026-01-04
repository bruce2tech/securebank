# securebank/modules/data/advanced_processor.py
"""
Advanced data processing pipeline for SecureBank fraud detection system.
Provides sophisticated feature engineering, quality validation, and drift detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings


class FeatureEngineer:
    """
    Advanced feature engineering for fraud detection datasets.
    """
    
    def __init__(self):
        self.feature_history = []
        self.scalers = {}
        
    def create_temporal_features(self, df: pd.DataFrame, 
                                datetime_col: str = 'trans_date_trans_time') -> pd.DataFrame:
        """
        Create comprehensive temporal features from transaction timestamps.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with datetime column.
        datetime_col : str
            Name of datetime column.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with additional temporal features.
        """
        result = df.copy()
        
        # Ensure datetime column exists and is datetime type
        if datetime_col not in result.columns:
            raise ValueError(f"Datetime column '{datetime_col}' not found in dataframe")
        
        result[datetime_col] = pd.to_datetime(result[datetime_col])
        dt_col = result[datetime_col]
        
        # Basic temporal features
        result['transaction_hour'] = dt_col.dt.hour
        result['transaction_day'] = dt_col.dt.day
        result['transaction_month'] = dt_col.dt.month
        result['transaction_year'] = dt_col.dt.year
        result['day_of_week'] = dt_col.dt.dayofweek
        result['day_of_year'] = dt_col.dt.dayofyear
        result['week_of_year'] = dt_col.dt.isocalendar().week
        result['quarter'] = dt_col.dt.quarter
        
        # Time period categorizations
        result['is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
        result['is_business_hours'] = ((dt_col.dt.hour >= 9) & (dt_col.dt.hour <= 17)).astype(int)
        result['is_late_night'] = ((dt_col.dt.hour >= 23) | (dt_col.dt.hour <= 5)).astype(int)
        result['is_rush_hour'] = ((dt_col.dt.hour.isin([7, 8, 17, 18]))).astype(int)
        
        # Cyclical encoding for temporal features
        result['hour_sin'] = np.sin(2 * np.pi * dt_col.dt.hour / 24)
        result['hour_cos'] = np.cos(2 * np.pi * dt_col.dt.hour / 24)
        result['day_sin'] = np.sin(2 * np.pi * dt_col.dt.dayofweek / 7)
        result['day_cos'] = np.cos(2 * np.pi * dt_col.dt.dayofweek / 7)
        result['month_sin'] = np.sin(2 * np.pi * dt_col.dt.month / 12)
        result['month_cos'] = np.cos(2 * np.pi * dt_col.dt.month / 12)
        
        return result
    
    def create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced transaction-based features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with transaction data.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with additional transaction features.
        """
        result = df.copy()
        
        if 'amt' in result.columns:
            # Amount-based features
            result['amt_log'] = np.log1p(result['amt'])
            result['amt_sqrt'] = np.sqrt(result['amt'])
            result['amt_squared'] = result['amt'] ** 2
            
            # Amount categorization
            result['is_small_amount'] = (result['amt'] <= 10).astype(int)
            result['is_medium_amount'] = ((result['amt'] > 10) & (result['amt'] <= 100)).astype(int)
            result['is_large_amount'] = (result['amt'] > 100).astype(int)
            result['is_round_amount'] = (result['amt'] % 1 == 0).astype(int)
            
            # Statistical features
            result['amt_zscore'] = stats.zscore(result['amt'])
            result['amt_percentile'] = result['amt'].rank(pct=True)
        
        # Geographic features if available
        if all(col in result.columns for col in ['merch_lat', 'merch_long', 'lat', 'long']):
            # Distance between customer and merchant
            result['distance_km'] = self._calculate_distance(
                result['lat'], result['long'],
                result['merch_lat'], result['merch_long']
            )
            result['is_local_transaction'] = (result['distance_km'] <= 50).astype(int)
            result['is_distant_transaction'] = (result['distance_km'] > 200).astype(int)
        
        return result
    
    def create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer-level aggregated features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with customer and transaction data.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with additional customer features.
        """
        result = df.copy()
        
        if 'cc_num' not in result.columns:
            return result
        
        # Customer transaction aggregations
        customer_aggs = result.groupby('cc_num').agg({
            'amt': ['count', 'sum', 'mean', 'std', 'min', 'max'],
            'merchant': 'nunique',
            'category': 'nunique'
        }).round(3)
        
        # Flatten column names
        customer_aggs.columns = [f'customer_{col[0]}_{col[1]}' for col in customer_aggs.columns]
        customer_aggs = customer_aggs.add_prefix('').reset_index()
        
        # Merge back to main dataframe
        result = result.merge(customer_aggs, on='cc_num', how='left')
        
        # Customer behavior features
        if 'customer_amt_count' in result.columns:
            result['avg_transaction_amount'] = result['customer_amt_sum'] / result['customer_amt_count']
            result['transaction_amount_ratio'] = result['amt'] / (result['avg_transaction_amount'] + 1e-6)
            result['is_frequent_customer'] = (result['customer_amt_count'] > 10).astype(int)
        
        return result
    
    def create_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create merchant-level aggregated features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with merchant and transaction data.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with additional merchant features.
        """
        result = df.copy()
        
        if 'merchant' not in result.columns:
            return result
        
        # Merchant transaction aggregations
        merchant_aggs = result.groupby('merchant').agg({
            'amt': ['count', 'mean', 'std'],
            'cc_num': 'nunique'
        }).round(3)
        
        # Flatten column names
        merchant_aggs.columns = [f'merchant_{col[0]}_{col[1]}' for col in merchant_aggs.columns]
        merchant_aggs = merchant_aggs.add_prefix('').reset_index()
        
        # Merge back to main dataframe
        result = result.merge(merchant_aggs, on='merchant', how='left')
        
        # Merchant behavior features
        if 'merchant_amt_mean' in result.columns:
            result['merchant_amount_deviation'] = (result['amt'] - result['merchant_amt_mean']) / (result['merchant_amt_std'] + 1e-6)
            result['is_popular_merchant'] = (result['merchant_cc_num_nunique'] > 100).astype(int)
        
        return result
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with interaction features.
        """
        result = df.copy()
        
        # Time-amount interactions
        if all(col in result.columns for col in ['transaction_hour', 'amt']):
            result['hour_amount_interaction'] = result['transaction_hour'] * result['amt_log'] if 'amt_log' in result.columns else result['transaction_hour'] * np.log1p(result['amt'])
        
        # Weekend-amount interaction
        if all(col in result.columns for col in ['is_weekend', 'amt']):
            result['weekend_amount_interaction'] = result['is_weekend'] * result['amt']
        
        # Customer-merchant interaction
        if all(col in result.columns for col in ['customer_amt_count', 'merchant_amt_count']):
            result['customer_merchant_interaction'] = result['customer_amt_count'] * result['merchant_amt_count']
        
        return result
    
    def _calculate_distance(self, lat1: pd.Series, lon1: pd.Series, 
                           lat2: pd.Series, lon2: pd.Series) -> pd.Series:
        """
        Calculate haversine distance between two sets of coordinates.
        
        Parameters
        ----------
        lat1, lon1 : pd.Series
            First set of coordinates.
        lat2, lon2 : pd.Series
            Second set of coordinates.
            
        Returns
        -------
        pd.Series
            Distance in kilometers.
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r


class DataQualityValidator:
    """
    Comprehensive data quality validation and scoring system.
    """
    
    def __init__(self):
        self.validation_rules = {}
        self.quality_thresholds = {
            'completeness': 0.95,
            'consistency': 0.90,
            'validity': 0.95,
            'accuracy': 0.85
        }
    
    def validate_dataset(self, df: pd.DataFrame, 
                        target_col: str = 'is_fraud') -> Dict[str, Any]:
        """
        Perform comprehensive data quality validation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset to validate.
        target_col : str
            Target column name.
            
        Returns
        -------
        dict
            Comprehensive quality report.
        """
        quality_report = {
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            'completeness': self._check_completeness(df),
            'consistency': self._check_consistency(df),
            'validity': self._check_validity(df),
            'accuracy': self._check_accuracy(df, target_col),
            'outliers': self._detect_outliers(df),
            'data_types': self._validate_data_types(df),
            'overall_score': 0.0,
            'recommendations': [],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Calculate overall quality score
        quality_report['overall_score'] = self._calculate_overall_score(quality_report)
        
        # Generate recommendations
        quality_report['recommendations'] = self._generate_recommendations(quality_report)
        
        return quality_report
    
    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness (missing values)."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df) * 100).round(2)
        
        completeness_score = 1 - (missing_counts.sum() / df.size)
        
        return {
            'score': round(completeness_score, 4),
            'missing_by_column': missing_percentages[missing_percentages > 0].to_dict(),
            'total_missing_cells': int(missing_counts.sum()),
            'columns_with_missing': len(missing_percentages[missing_percentages > 0])
        }
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency across related fields."""
        consistency_issues = []
        consistency_score = 1.0
        
        # Check for duplicate transactions
        if 'trans_num' in df.columns:
            duplicates = df['trans_num'].duplicated().sum()
            if duplicates > 0:
                consistency_issues.append(f"{duplicates} duplicate transaction numbers")
                consistency_score -= 0.1
        
        # Check amount consistency (negative amounts)
        if 'amt' in df.columns:
            negative_amounts = (df['amt'] < 0).sum()
            if negative_amounts > 0:
                consistency_issues.append(f"{negative_amounts} negative transaction amounts")
                consistency_score -= 0.05
        
        # Check date consistency
        if 'trans_date_trans_time' in df.columns:
            try:
                dt_col = pd.to_datetime(df['trans_date_trans_time'])
                future_dates = (dt_col > datetime.now()).sum()
                if future_dates > 0:
                    consistency_issues.append(f"{future_dates} future transaction dates")
                    consistency_score -= 0.05
            except:
                consistency_issues.append("Date format inconsistency detected")
                consistency_score -= 0.1
        
        return {
            'score': max(0, consistency_score),
            'issues': consistency_issues,
            'issue_count': len(consistency_issues)
        }
    
    def _check_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data validity (format and range constraints)."""
        validity_issues = []
        validity_score = 1.0
        
        # Check credit card numbers
        if 'cc_num' in df.columns:
            invalid_cc = df['cc_num'].astype(str).str.len() != 16
            invalid_count = invalid_cc.sum()
            if invalid_count > 0:
                validity_issues.append(f"{invalid_count} invalid credit card number formats")
                validity_score -= 0.1
        
        # Check email formats if present
        if 'email' in df.columns:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            invalid_emails = ~df['email'].str.match(email_pattern, na=False)
            invalid_count = invalid_emails.sum()
            if invalid_count > 0:
                validity_issues.append(f"{invalid_count} invalid email formats")
                validity_score -= 0.05
        
        # Check coordinate ranges
        for col, range_check in [('lat', (-90, 90)), ('long', (-180, 180)), 
                                ('merch_lat', (-90, 90)), ('merch_long', (-180, 180))]:
            if col in df.columns:
                out_of_range = ~df[col].between(range_check[0], range_check[1])
                invalid_count = out_of_range.sum()
                if invalid_count > 0:
                    validity_issues.append(f"{invalid_count} invalid {col} coordinates")
                    validity_score -= 0.02
        
        return {
            'score': max(0, validity_score),
            'issues': validity_issues,
            'issue_count': len(validity_issues)
        }
    
    def _check_accuracy(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Check data accuracy using statistical methods."""
        accuracy_issues = []
        accuracy_score = 1.0
        
        # Check target distribution
        if target_col in df.columns:
            target_dist = df[target_col].value_counts(normalize=True)
            if len(target_dist) == 2:
                fraud_rate = target_dist.get(1, 0)
                if fraud_rate < 0.001 or fraud_rate > 0.5:
                    accuracy_issues.append(f"Unusual fraud rate: {fraud_rate:.3f}")
                    accuracy_score -= 0.1
        
        # Check for impossible values
        if 'amt' in df.columns:
            zero_amounts = (df['amt'] == 0).sum()
            if zero_amounts > len(df) * 0.01:  # More than 1% zero amounts
                accuracy_issues.append(f"High number of zero-amount transactions: {zero_amounts}")
                accuracy_score -= 0.05
        
        return {
            'score': max(0, accuracy_score),
            'issues': accuracy_issues,
            'issue_count': len(accuracy_issues)
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numerical_cols:
            if col in df.columns and not df[col].isnull().all():
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_percentage = (outliers / len(df)) * 100
                
                outlier_info[col] = {
                    'count': int(outliers),
                    'percentage': round(outlier_percentage, 2),
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
        
        return outlier_info
    
    def _validate_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate and suggest appropriate data types."""
        type_recommendations = {}
        
        for col in df.columns:
            current_type = str(df[col].dtype)
            
            # Check if object columns should be categorical
            if current_type == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    type_recommendations[col] = {
                        'current': current_type,
                        'recommended': 'category',
                        'reason': f'Low cardinality: {df[col].nunique()} unique values'
                    }
            
            # Check if numerical columns have appropriate precision
            elif 'float' in current_type:
                if df[col].apply(lambda x: x.is_integer() if pd.notnull(x) else True).all():
                    type_recommendations[col] = {
                        'current': current_type,
                        'recommended': 'int64',
                        'reason': 'All values are integers'
                    }
        
        return type_recommendations
    
    def _calculate_overall_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        weights = {
            'completeness': 0.3,
            'consistency': 0.25,
            'validity': 0.25,
            'accuracy': 0.2
        }
        
        weighted_score = sum(
            quality_report[metric]['score'] * weight
            for metric, weight in weights.items()
        )
        
        return round(weighted_score, 4)
    
    def _generate_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on quality assessment."""
        recommendations = []
        
        # Completeness recommendations
        if quality_report['completeness']['score'] < self.quality_thresholds['completeness']:
            recommendations.append("Consider implementing advanced imputation strategies for missing values")
        
        # Consistency recommendations
        if quality_report['consistency']['issue_count'] > 0:
            recommendations.append("Address data consistency issues before model training")
        
        # Validity recommendations
        if quality_report['validity']['issue_count'] > 0:
            recommendations.append("Implement data validation rules at ingestion time")
        
        # Outlier recommendations
        outlier_columns = [col for col, info in quality_report['outliers'].items() 
                          if info['percentage'] > 5]
        if outlier_columns:
            recommendations.append(f"Consider outlier treatment for columns: {', '.join(outlier_columns)}")
        
        # Overall score recommendations
        if quality_report['overall_score'] < 0.8:
            recommendations.append("Dataset quality is below recommended threshold - consider data cleaning")
        
        return recommendations


class OutlierDetector:
    """
    Advanced outlier detection using multiple methods.
    """
    
    def __init__(self):
        self.detection_methods = ['iqr', 'zscore', 'isolation_forest']
        self.contamination_rate = 0.1
    
    def detect_outliers(self, df: pd.DataFrame, 
                       methods: List[str] = None,
                       numerical_only: bool = True) -> Dict[str, Any]:
        """
        Detect outliers using multiple methods.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        methods : list, optional
            Methods to use for outlier detection.
        numerical_only : bool
            Whether to only analyze numerical columns.
            
        Returns
        -------
        dict
            Outlier detection results.
        """
        if methods is None:
            methods = self.detection_methods
        
        if numerical_only:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            analysis_df = df[numeric_cols]
        else:
            analysis_df = df
        
        results = {
            'methods_used': methods,
            'outlier_detection': {},
            'summary': {}
        }
        
        for method in methods:
            if method == 'iqr':
                results['outlier_detection'][method] = self._iqr_outliers(analysis_df)
            elif method == 'zscore':
                results['outlier_detection'][method] = self._zscore_outliers(analysis_df)
            elif method == 'isolation_forest':
                results['outlier_detection'][method] = self._isolation_forest_outliers(analysis_df)
        
        # Create summary
        results['summary'] = self._summarize_outlier_detection(results['outlier_detection'])
        
        return results
    
    def _iqr_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        outliers_by_column = {}
        
        for col in df.columns:
            if df[col].dtype in [np.number]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_by_column[col] = {
                    'count': int(outlier_mask.sum()),
                    'percentage': round((outlier_mask.sum() / len(df)) * 100, 2),
                    'indices': outlier_mask[outlier_mask].index.tolist()
                }
        
        return outliers_by_column
    
    def _zscore_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        outliers_by_column = {}
        
        for col in df.columns:
            if df[col].dtype in [np.number]:
                z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
                outlier_mask = z_scores > threshold
                
                outliers_by_column[col] = {
                    'count': int(outlier_mask.sum()),
                    'percentage': round((outlier_mask.sum() / len(df)) * 100, 2),
                    'indices': outlier_mask[outlier_mask].index.tolist()
                }
        
        return outliers_by_column
    
    def _isolation_forest_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest."""
        numeric_df = df.select_dtypes(include=[np.number]).fillna(df.mean())
        
        if len(numeric_df.columns) == 0:
            return {'error': 'No numerical columns for Isolation Forest'}
        
        iso_forest = IsolationForest(
            contamination=self.contamination_rate,
            random_state=42,
            n_estimators=100
        )
        
        outlier_labels = iso_forest.fit_predict(numeric_df)
        outlier_mask = outlier_labels == -1
        
        return {
            'global_outliers': {
                'count': int(outlier_mask.sum()),
                'percentage': round((outlier_mask.sum() / len(df)) * 100, 2),
                'indices': outlier_mask[outlier_mask].index.tolist()
            }
        }
    
    def _summarize_outlier_detection(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize outlier detection across methods."""
        summary = {
            'total_methods': len(detection_results),
            'consensus_outliers': [],
            'method_agreement': {}
        }
        
        # Find consensus outliers (detected by multiple methods)
        all_outlier_indices = []
        for method, results in detection_results.items():
            if method == 'isolation_forest':
                if 'global_outliers' in results:
                    all_outlier_indices.extend(results['global_outliers']['indices'])
            else:
                for col_results in results.values():
                    all_outlier_indices.extend(col_results['indices'])
        
        # Count frequency of each index
        from collections import Counter
        index_counts = Counter(all_outlier_indices)
        
        # Find indices detected by multiple methods
        consensus_threshold = max(1, len(detection_results) // 2)
        summary['consensus_outliers'] = [
            idx for idx, count in index_counts.items() 
            if count >= consensus_threshold
        ]
        
        summary['consensus_count'] = len(summary['consensus_outliers'])
        
        return summary