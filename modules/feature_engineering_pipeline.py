"""
Advanced Feature Engineering Pipeline
Part of SecureBank Phase 4: Enhanced Dataset Generation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Advanced feature engineering for fraud detection with behavioral analytics"""
    
    def __init__(self):
        self.feature_stats = {}
        self.engineered_features = []
        
    def engineer_features(self, df: pd.DataFrame, create_behavioral: bool = True, 
                         create_temporal: bool = True, create_spatial: bool = True,
                         create_statistical: bool = True, is_prediction: bool = False) -> pd.DataFrame:
        """
        Comprehensive feature engineering pipeline
        
        Args:
            df: Input dataframe
            create_behavioral: Create behavioral/pattern features
            create_temporal: Create time-based features
            create_spatial: Create location-based features
            create_statistical: Create statistical aggregation features
            is_prediction: Whether this is for prediction (affects target column handling)
        """
        
        print("🔧 Starting advanced feature engineering...")
        df_enhanced = df.copy()
        
        # Ensure datetime conversion
        if 'trans_date_trans_time' in df_enhanced.columns:
            df_enhanced['trans_date_trans_time'] = pd.to_datetime(df_enhanced['trans_date_trans_time'])
        
        feature_count = len(df_enhanced.columns)
        
        if create_temporal:
            df_enhanced = self._create_temporal_features(df_enhanced)
            print(f"   ⏰ Added {len(df_enhanced.columns) - feature_count} temporal features")
            feature_count = len(df_enhanced.columns)
        
        if create_spatial:
            df_enhanced = self._create_spatial_features(df_enhanced)
            print(f"   🌍 Added {len(df_enhanced.columns) - feature_count} spatial features")
            feature_count = len(df_enhanced.columns)
        
        if create_behavioral:
            df_enhanced = self._create_behavioral_features(df_enhanced, is_prediction)
            print(f"   🧠 Added {len(df_enhanced.columns) - feature_count} behavioral features")
            feature_count = len(df_enhanced.columns)
        
        if create_statistical:
            df_enhanced = self._create_statistical_features(df_enhanced, is_prediction)
            print(f"   📊 Added {len(df_enhanced.columns) - feature_count} statistical features")
        
        total_new_features = len(df_enhanced.columns) - len(df.columns)
        print(f"✅ Feature engineering completed! Added {total_new_features} new features")
        
        # Final data cleaning - handle any NaN values created during feature engineering
        print("🧽 Cleaning engineered features...")
        
        # For prediction mode, be extra careful with data types
        if is_prediction:
            # Convert all object columns to numeric for prediction
            for col in df_enhanced.columns:
                if df_enhanced[col].dtype == 'object':
                    # Try to convert to numeric, otherwise encode categorically
                    try:
                        df_enhanced[col] = pd.to_numeric(df_enhanced[col], errors='coerce')
                    except:
                        # Simple encoding for categorical columns
                        if df_enhanced[col].nunique() <= 50:
                            # Label encode small categories
                            unique_vals = df_enhanced[col].unique()
                            value_map = {val: idx for idx, val in enumerate(unique_vals)}
                            df_enhanced[col] = df_enhanced[col].map(value_map)
                        else:
                            # Hash encode large categories
                            df_enhanced[col] = df_enhanced[col].astype(str).apply(hash).abs() % 1000
        
        # Fill NaN values with appropriate defaults
        numeric_columns = df_enhanced.select_dtypes(include=[np.number]).columns
        categorical_columns = df_enhanced.select_dtypes(include=['object']).columns
        
        # Fill numeric NaN with 0
        df_enhanced[numeric_columns] = df_enhanced[numeric_columns].fillna(0)
        
        # Fill categorical NaN with 'Unknown' or encode if prediction mode
        if is_prediction:
            df_enhanced[categorical_columns] = df_enhanced[categorical_columns].fillna(0)
        else:
            df_enhanced[categorical_columns] = df_enhanced[categorical_columns].fillna('Unknown')
        
        # Ensure no infinite values
        df_enhanced = df_enhanced.replace([np.inf, -np.inf], 0)
        
        # For prediction, ensure all columns are numeric
        if is_prediction:
            for col in df_enhanced.columns:
                if df_enhanced[col].dtype == 'object':
                    df_enhanced[col] = 0  # Final fallback
            df_enhanced = df_enhanced.select_dtypes(include=[np.number])
        
        print(f"✅ Data cleaning completed. Final shape: {df_enhanced.shape}")
        
        # Store feature information
        self.engineered_features = [col for col in df_enhanced.columns if col not in df.columns]
        
        return df_enhanced
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive temporal features"""
        
        if 'trans_date_trans_time' not in df.columns:
            return df
        
        # Normal statistical feature creation for training data
        # Merchant-level statistics
        agg_dict_merchant = {'amt': ['count', 'mean', 'std', 'median']}
        if 'is_fraud' in df.columns:
            agg_dict_merchant['is_fraud'] = 'mean'
        
        merchant_stats = df.groupby('merchant').agg(agg_dict_merchant).round(4)
        
        merchant_stats.columns = ['_'.join(col).strip() for col in merchant_stats.columns]
        merchant_stats = merchant_stats.add_prefix('merchant_')
        
        # Check if these columns already exist to avoid merge conflicts
        existing_merchant_cols = [col for col in merchant_stats.columns if col in df.columns]
        if existing_merchant_cols:
            print(f"⚠️ Dropping existing merchant columns to avoid duplicates: {existing_merchant_cols}")
            df = df.drop(columns=existing_merchant_cols)
        
        df = df.merge(merchant_stats, left_on='merchant', right_index=True, how='left')
        
        # Category-level statistics
        agg_dict_category = {'amt': ['count', 'mean', 'std', 'median']}
        if 'is_fraud' in df.columns:
            agg_dict_category['is_fraud'] = 'mean'
        
        category_stats = df.groupby('category').agg(agg_dict_category).round(4)
        
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
        category_stats = category_stats.add_prefix('category_')
        
        # Check if these columns already exist to avoid merge conflicts
        existing_category_cols = [col for col in category_stats.columns if col in df.columns]
        if existing_category_cols:
            print(f"⚠️ Dropping existing category columns to avoid duplicates: {existing_category_cols}")
            df = df.drop(columns=existing_category_cols)
        
        df = df.merge(category_stats, left_on='category', right_index=True, how='left')
        
        # Time-based rolling statistics
        if 'trans_date_trans_time' in df.columns and 'cc_num' in df.columns:
            # Sort by customer and time
            df_sorted = df.sort_values(['cc_num', 'trans_date_trans_time']).copy()
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df_sorted['trans_date_trans_time']):
                df_sorted['trans_date_trans_time'] = pd.to_datetime(df_sorted['trans_date_trans_time'])
            
            # Set datetime as index for rolling operations
            df_sorted = df_sorted.set_index('trans_date_trans_time')
            
            # Rolling window features (last 7 days) - use integer window instead of time-based
            # Calculate approximate window size (assume average 2 transactions per day per customer)
            rolling_window = min(14, max(3, len(df_sorted) // 100))  # Adaptive window size
            
            df_sorted['rolling_txn_count_7d'] = df_sorted.groupby('cc_num')['amt'].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1).count()
            )
            df_sorted['rolling_avg_amount_7d'] = df_sorted.groupby('cc_num')['amt'].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1).mean()
            )
            df_sorted['rolling_std_amount_7d'] = df_sorted.groupby('cc_num')['amt'].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1).std().fillna(0)
            )
            
            # Reset index and merge back
            df_sorted = df_sorted.reset_index()
            rolling_cols = ['rolling_txn_count_7d', 'rolling_avg_amount_7d', 'rolling_std_amount_7d']
            
            # Check if rolling columns already exist to avoid merge conflicts
            existing_rolling_cols = [col for col in rolling_cols if col in df.columns]
            if existing_rolling_cols:
                print(f"⚠️ Dropping existing rolling columns to avoid duplicates: {existing_rolling_cols}")
                df = df.drop(columns=existing_rolling_cols)
            
            # Merge back maintaining original order
            df = df.merge(
                df_sorted[['cc_num', 'trans_date_trans_time'] + rolling_cols], 
                on=['cc_num', 'trans_date_trans_time'], 
                how='left'
            )
        
        # Cross-feature interactions
        if 'transaction_hour' in df.columns:
            df['hour_category_interaction'] = df['transaction_hour'].astype(str) + '_' + df['category']
            df['hour_amount_interaction'] = df['transaction_hour'] * df['amt']
        
        if 'is_weekend' in df.columns:
            df['weekend_amount_interaction'] = df['is_weekend'] * df['amt']
        
        return df
        
        dt_col = df['trans_date_trans_time']
        
        # Basic temporal features
        df['transaction_hour'] = dt_col.dt.hour
        df['transaction_day_of_week'] = dt_col.dt.dayofweek
        df['transaction_day_of_month'] = dt_col.dt.day
        df['transaction_month'] = dt_col.dt.month
        df['transaction_year'] = dt_col.dt.year
        df['transaction_quarter'] = dt_col.dt.quarter
        
        # Advanced temporal features
        df['is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
        df['is_holiday_season'] = ((dt_col.dt.month == 12) | (dt_col.dt.month == 1)).astype(int)
        df['is_business_hour'] = ((dt_col.dt.hour >= 9) & (dt_col.dt.hour <= 17)).astype(int)
        df['is_late_night'] = ((dt_col.dt.hour >= 22) | (dt_col.dt.hour <= 5)).astype(int)
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['transaction_day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['transaction_day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['transaction_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['transaction_month'] / 12)
        
        # Time since features (requires sorting by time first)
        df_sorted = df.sort_values('trans_date_trans_time')
        df_sorted['time_since_last_transaction'] = df_sorted['trans_date_trans_time'].diff().dt.total_seconds().fillna(0)
        df_sorted['days_since_first_transaction'] = (df_sorted['trans_date_trans_time'] - df_sorted['trans_date_trans_time'].min()).dt.days
        
        # Merge back maintaining original order
        df = df.merge(df_sorted[['time_since_last_transaction', 'days_since_first_transaction']], 
                     left_index=True, right_index=True, how='left')
        
        return df
    
    def _create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based features"""
        
        if 'merch_lat' not in df.columns or 'merch_long' not in df.columns:
            return df
        
        # Distance from common reference points
        # Major US financial centers
        reference_points = {
            'NYC': (40.7128, -74.0060),
            'Chicago': (41.8781, -87.6298),
            'LA': (34.0522, -118.2437),
            'Houston': (29.7604, -95.3698),
            'Atlanta': (33.7490, -84.3880)
        }
        
        for city, (ref_lat, ref_lon) in reference_points.items():
            df[f'distance_from_{city.lower()}'] = self._haversine_distance(
                df['merch_lat'], df['merch_long'], ref_lat, ref_lon
            )
        
        # Geographic clustering features
        df['lat_rounded'] = np.round(df['merch_lat'], 1)
        df['long_rounded'] = np.round(df['merch_long'], 1)
        df['geo_cluster'] = df['lat_rounded'].astype(str) + '_' + df['long_rounded'].astype(str)
        
        # Distance from center of merchant distribution
        center_lat = df['merch_lat'].median()
        center_long = df['merch_long'].median()
        df['distance_from_center'] = self._haversine_distance(
            df['merch_lat'], df['merch_long'], center_lat, center_long
        )
        
        # Outlier detection for locations
        lat_q75, lat_q25 = df['merch_lat'].quantile([0.75, 0.25])
        long_q75, long_q25 = df['merch_long'].quantile([0.75, 0.25])
        lat_iqr = lat_q75 - lat_q25
        long_iqr = long_q75 - long_q25
        
        df['is_location_outlier'] = (
            (df['merch_lat'] < (lat_q25 - 1.5 * lat_iqr)) |
            (df['merch_lat'] > (lat_q75 + 1.5 * lat_iqr)) |
            (df['merch_long'] < (long_q25 - 1.5 * long_iqr)) |
            (df['merch_long'] > (long_q75 + 1.5 * long_iqr))
        ).astype(int)
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame, is_prediction: bool = False) -> pd.DataFrame:
        """Create customer behavioral pattern features"""
        
        if 'cc_num' not in df.columns:
            return df
        
        # For prediction, we need to use simpler features since we don't have historical data
        if is_prediction and len(df) == 1:
            # For single transaction prediction, create minimal behavioral features
            df['customer_avg_amount'] = df['amt']  # Use current amount as average
            df['amount_ratio_to_avg'] = 1.0  # Ratio to self is 1
            df['amount_zscore'] = 0.0  # No z-score without history
            df['is_amount_outlier'] = 0  # Not an outlier by definition
            df['customer_favorite_merchant'] = df['merchant']
            df['customer_favorite_category'] = df['category']
            df['is_familiar_merchant'] = 1
            df['is_familiar_category'] = 1
            
            # Add other customer features with defaults
            df['customer_amt_count'] = 1
            df['customer_amt_mean'] = df['amt']
            df['customer_amt_std'] = 0.0
            df['customer_amt_min'] = df['amt']
            df['customer_amt_max'] = df['amt']
            df['customer_amt_sum'] = df['amt']
            df['customer_merchant_nunique'] = 1
            df['customer_category_nunique'] = 1
            df['customer_daily_txn_mean'] = 1.0
            df['customer_daily_txn_std'] = 0.0
            
            return df
        
        # Customer-level aggregations (avoid target column references during prediction)
        try:
            agg_dict = {
                'amt': ['count', 'mean', 'std', 'min', 'max', 'sum'],
                'merchant': 'nunique',
                'category': 'nunique'
            }
            
            customer_stats = df.groupby('cc_num').agg(agg_dict).round(4)
            
            # Flatten column names
            customer_stats.columns = ['_'.join(col).strip() for col in customer_stats.columns]
            customer_stats = customer_stats.add_prefix('customer_')
            
            # Merge back to main dataset
            df = df.merge(customer_stats, left_on='cc_num', right_index=True, how='left')
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create customer aggregations: {str(e)}")
            # Create default customer features
            df['customer_amt_count'] = 1
            df['customer_amt_mean'] = df['amt']
            df['customer_amt_std'] = 0.0
            df['customer_amt_min'] = df['amt']
            df['customer_amt_max'] = df['amt']
            df['customer_amt_sum'] = df['amt']
            df['customer_merchant_nunique'] = 1
            df['customer_category_nunique'] = 1
        
        # Transaction frequency features
        if 'trans_date_trans_time' in df.columns:
            try:
                # Daily transaction patterns
                df['date_only'] = df['trans_date_trans_time'].dt.date
                daily_transactions = df.groupby(['cc_num', 'date_only']).size().reset_index(name='daily_transaction_count')
                daily_stats = daily_transactions.groupby('cc_num')['daily_transaction_count'].agg(['mean', 'std']).add_prefix('customer_daily_txn_')
                df = df.merge(daily_stats, left_on='cc_num', right_index=True, how='left')
            except Exception as e:
                print(f"⚠️ Warning: Could not create daily transaction features: {str(e)}")
                # Create default daily transaction features
                df['customer_daily_txn_mean'] = 1.0
                df['customer_daily_txn_std'] = 0.0
        
        # Merchant and category behavior
        try:
            merchant_stats = df.groupby('cc_num')['merchant'].apply(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
            ).reset_index()
            merchant_stats.columns = ['cc_num', 'customer_favorite_merchant']
            df = df.merge(merchant_stats, on='cc_num', how='left')
            
            category_stats = df.groupby('cc_num')['category'].apply(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
            ).reset_index()
            category_stats.columns = ['cc_num', 'customer_favorite_category']
            df = df.merge(category_stats, on='cc_num', how='left')
            
            # Merchant familiarity (only if columns exist)
            if 'customer_favorite_merchant' in df.columns:
                df['is_familiar_merchant'] = (df['merchant'] == df['customer_favorite_merchant']).astype(int)
            if 'customer_favorite_category' in df.columns:
                df['is_familiar_category'] = (df['category'] == df['customer_favorite_category']).astype(int)
                
        except Exception as e:
            print(f"⚠️ Warning: Could not create favorite merchant/category features: {str(e)}")
            # Create default values
            df['customer_favorite_merchant'] = 'Unknown'
            df['customer_favorite_category'] = 'Unknown'
            df['is_familiar_merchant'] = 0
            df['is_familiar_category'] = 0

        # Transaction amount patterns
        df['amount_zscore'] = df.groupby('cc_num')['amt'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
        df['is_amount_outlier'] = (np.abs(df['amount_zscore']) > 2).astype(int)
        
        # Customer spending velocity
        df['customer_avg_amount'] = df.groupby('cc_num')['amt'].transform('mean')
        df['amount_ratio_to_avg'] = df['amt'] / (df['customer_avg_amount'] + 1e-8)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame, is_prediction: bool = False) -> pd.DataFrame:
        """Create statistical aggregation features"""
        
        # For prediction, create simplified statistical features
        if is_prediction and len(df) == 1:
            # Create default statistical features for single prediction
            df['merchant_amt_count'] = 1
            df['merchant_amt_mean'] = df['amt'].iloc[0]
            df['merchant_amt_std'] = 0.0
            df['merchant_amt_median'] = df['amt'].iloc[0]
            
            df['category_amt_count'] = 1  
            df['category_amt_mean'] = df['amt'].iloc[0]
            df['category_amt_std'] = 0.0
            df['category_amt_median'] = df['amt'].iloc[0]
            
            # Set fraud rate features to neutral values (no historical data)
            if 'is_fraud' not in df.columns:
                df['merchant_is_fraud_mean'] = 0.0
                df['category_is_fraud_mean'] = 0.0
            
            # Rolling features with defaults for single transaction
            df['rolling_txn_count_7d'] = 1
            df['rolling_avg_amount_7d'] = df['amt'].iloc[0]
            df['rolling_std_amount_7d'] = 0.0
            
            # Cross-feature interactions
            if 'transaction_hour' in df.columns:
                df['hour_category_interaction'] = f"{df['transaction_hour'].iloc[0]}_{df['category'].iloc[0]}"
                df['hour_amount_interaction'] = df['transaction_hour'].iloc[0] * df['amt'].iloc[0]
            
            if 'is_weekend' in df.columns:
                df['weekend_amount_interaction'] = df['is_weekend'].iloc[0] * df['amt'].iloc[0]
            
            return df
        
        # Merchant-level statistics
        merchant_stats = df.groupby('merchant').agg({
            'amt': ['count', 'mean', 'std', 'median'],
            'is_fraud': 'mean' if 'is_fraud' in df.columns else 'size'
        }).round(4)
        
        merchant_stats.columns = ['_'.join(col).strip() for col in merchant_stats.columns]
        merchant_stats = merchant_stats.add_prefix('merchant_')
        df = df.merge(merchant_stats, left_on='merchant', right_index=True, how='left')
        
        # Category-level statistics
        category_stats = df.groupby('category').agg({
            'amt': ['count', 'mean', 'std', 'median'],
            'is_fraud': 'mean' if 'is_fraud' in df.columns else 'size'
        }).round(4)
        
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
        category_stats = category_stats.add_prefix('category_')
        df = df.merge(category_stats, left_on='category', right_index=True, how='left')
        
        # Time-based rolling statistics
        if 'trans_date_trans_time' in df.columns and 'cc_num' in df.columns:
            # Sort by customer and time
            df_sorted = df.sort_values(['cc_num', 'trans_date_trans_time']).copy()
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df_sorted['trans_date_trans_time']):
                df_sorted['trans_date_trans_time'] = pd.to_datetime(df_sorted['trans_date_trans_time'])
            
            # Set datetime as index for rolling operations
            df_sorted = df_sorted.set_index('trans_date_trans_time')
            
            # Rolling window features (last 7 days) - use integer window instead of time-based
            # Calculate approximate window size (assume average 2 transactions per day per customer)
            rolling_window = min(14, max(3, len(df_sorted) // 100))  # Adaptive window size
            
            df_sorted['rolling_txn_count_7d'] = df_sorted.groupby('cc_num')['amt'].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1).count()
            )
            df_sorted['rolling_avg_amount_7d'] = df_sorted.groupby('cc_num')['amt'].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1).mean()
            )
            df_sorted['rolling_std_amount_7d'] = df_sorted.groupby('cc_num')['amt'].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1).std().fillna(0)
            )
            
            # Reset index and merge back
            df_sorted = df_sorted.reset_index()
            rolling_cols = ['rolling_txn_count_7d', 'rolling_avg_amount_7d', 'rolling_std_amount_7d']
            
            # Merge back maintaining original order
            df = df.merge(
                df_sorted[['cc_num', 'trans_date_trans_time'] + rolling_cols], 
                on=['cc_num', 'trans_date_trans_time'], 
                how='left'
            )
        
        # Cross-feature interactions
        if 'transaction_hour' in df.columns:
            df['hour_category_interaction'] = df['transaction_hour'].astype(str) + '_' + df['category']
            df['hour_amount_interaction'] = df['transaction_hour'] * df['amt']
        
        if 'is_weekend' in df.columns:
            df['weekend_amount_interaction'] = df['is_weekend'] * df['amt']
        
        return df
    
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate haversine distance between coordinates"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        return distance
    
    def get_feature_importance_analysis(self, df: pd.DataFrame, target_col: str = 'is_fraud') -> Dict[str, Any]:
        """Analyze feature importance of engineered features"""
        
        # Skip feature importance analysis if target column doesn't exist (prediction scenario)
        if target_col not in df.columns:
            print(f"⚠️ Skipping feature importance analysis - target column '{target_col}' not found")
            return {
                "message": f"Feature importance analysis skipped - target column '{target_col}' not available",
                "total_features": len(df.columns),
                "engineered_features_count": len(self.engineered_features)
            }
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare features for analysis
        feature_df = df.select_dtypes(include=[np.number]).copy()
        
        # Handle categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        
        for col in categorical_cols:
            if col != target_col:
                try:
                    feature_df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                except:
                    pass
        
        # Remove target from features
        if target_col in feature_df.columns:
            X = feature_df.drop(columns=[target_col])
            y = feature_df[target_col]
        else:
            return {"error": "Target column not in numeric features"}
        
        # Train simple model for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X.fillna(0), y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Identify engineered features
        engineered_importance = importance_df[
            importance_df['feature'].isin(self.engineered_features)
        ].head(10)
        
        return {
            'top_overall_features': importance_df.head(15).to_dict('records'),
            'top_engineered_features': engineered_importance.to_dict('records'),
            'total_features': len(X.columns),
            'engineered_features_count': len(self.engineered_features)
        }
    
    def generate_feature_report(self, df: pd.DataFrame, save_path: str = None) -> str:
        """Generate comprehensive feature engineering report"""
        
        original_features = [col for col in df.columns if col not in self.engineered_features]
        
        report = f"""
# 🔧 FEATURE ENGINEERING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 FEATURE SUMMARY
- **Original Features**: {len(original_features)}
- **Engineered Features**: {len(self.engineered_features)}
- **Total Features**: {len(df.columns)}
- **Enhancement Ratio**: {len(self.engineered_features) / len(original_features):.1f}x

## 🆕 NEW FEATURES CREATED

### ⏰ Temporal Features
"""
        
        temporal_features = [f for f in self.engineered_features if any(x in f.lower() for x in ['hour', 'day', 'month', 'time', 'weekend', 'business', 'night', 'sin', 'cos'])]
        for feature in temporal_features:
            report += f"- {feature}\n"
        
        report += f"""
### 🌍 Spatial Features
"""
        spatial_features = [f for f in self.engineered_features if any(x in f.lower() for x in ['distance', 'lat', 'long', 'geo', 'location'])]
        for feature in spatial_features:
            report += f"- {feature}\n"
        
        report += f"""
### 🧠 Behavioral Features
"""
        behavioral_features = [f for f in self.engineered_features if any(x in f.lower() for x in ['customer', 'favorite', 'familiar', 'outlier', 'zscore', 'ratio'])]
        for feature in behavioral_features:
            report += f"- {feature}\n"
        
        report += f"""
### 📊 Statistical Features
"""
        statistical_features = [f for f in self.engineered_features if any(x in f.lower() for x in ['merchant', 'category', 'rolling', 'interaction', 'mean', 'std'])]
        for feature in statistical_features:
            report += f"- {feature}\n"
        
        # Feature quality assessment
        missing_data = df[self.engineered_features].isnull().sum()
        problematic_features = missing_data[missing_data > 0]
        
        report += f"""
## 🔍 FEATURE QUALITY ASSESSMENT
- **Complete Features**: {len(self.engineered_features) - len(problematic_features)}
- **Features with Missing Data**: {len(problematic_features)}
"""
        
        if len(problematic_features) > 0:
            report += "\n### ⚠️ Features Requiring Attention:\n"
            for feature, missing_count in problematic_features.items():
                missing_pct = (missing_count / len(df)) * 100
                report += f"- {feature}: {missing_count} missing ({missing_pct:.1f}%)\n"
        
        report += f"""
## 🎯 FEATURE ENGINEERING IMPACT
- **Data Richness**: Increased by {len(self.engineered_features)} dimensions
- **Temporal Coverage**: {len(temporal_features)} time-based features
- **Behavioral Analytics**: {len(behavioral_features)} customer pattern features
- **Spatial Intelligence**: {len(spatial_features)} location-based features
- **Statistical Depth**: {len(statistical_features)} aggregation features

## 💡 RECOMMENDATIONS
1. Monitor feature stability in production
2. Implement feature importance tracking
3. Consider feature selection for model optimization
4. Validate feature distributions across time periods
5. Implement automated feature quality monitoring

## 📈 NEXT STEPS
1. Perform feature selection analysis
2. Validate feature stability over time
3. Implement feature monitoring in production
4. Consider additional domain-specific features
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"📄 Feature engineering report saved to: {save_path}")
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Create sample test data
    np.random.seed(42)
    
    sample_data = {
        'trans_date_trans_time': pd.date_range('2023-01-01', periods=1000, freq='2H'),
        'cc_num': np.random.choice(range(1000, 2000), 1000),
        'merchant': np.random.choice(['Store_A', 'Store_B', 'Store_C', 'Online_Shop', 'Gas_Station'], 1000),
        'category': np.random.choice(['grocery_pos', 'gas_transport', 'entertainment', 'food_dining'], 1000),
        'amt': np.random.lognormal(3, 1, 1000),
        'unix_time': [int(dt.timestamp()) for dt in pd.date_range('2023-01-01', periods=1000, freq='2H')],
        'merch_lat': np.random.uniform(25, 45, 1000),  # US latitude range
        'merch_long': np.random.uniform(-125, -65, 1000),  # US longitude range
        'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Test feature engineering
    engineer = AdvancedFeatureEngineer()
    enhanced_df = engineer.engineer_features(sample_df)
    
    print(f"\nOriginal features: {len(sample_df.columns)}")
    print(f"Enhanced features: {len(enhanced_df.columns)}")
    print(f"New features added: {len(enhanced_df.columns) - len(sample_df.columns)}")
    
    # Test feature importance analysis
    importance_analysis = engineer.get_feature_importance_analysis(enhanced_df)
    print(f"\nTop 5 features by importance:")
    for i, feature in enumerate(importance_analysis['top_overall_features'][:5]):
        print(f"{i+1}. {feature['feature']}: {feature['importance']:.4f}")
    
    # Generate report
    report = engineer.generate_feature_report(enhanced_df)
    print("\n" + "="*50)
    print(report)