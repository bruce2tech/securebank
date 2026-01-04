"""
Advanced Data Quality Validation and Scoring System
Part of SecureBank Phase 4: Enhanced Dataset Generation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DataQualityMetrics:
    """Data quality metrics for comprehensive assessment"""
    completeness_score: float
    consistency_score: float
    validity_score: float
    uniqueness_score: float
    timeliness_score: float
    overall_score: float
    issues_found: List[str]
    recommendations: List[str]
    
class AdvancedDataQualityValidator:
    """Advanced data quality validation with scoring and recommendations"""
    
    def __init__(self):
        self.quality_thresholds = {
            'completeness': 0.95,  # 95% data completeness required
            'consistency': 0.90,   # 90% consistency required
            'validity': 0.85,      # 85% validity required
            'uniqueness': 0.99,    # 99% uniqueness for key fields
            'timeliness': 0.90     # 90% timeliness score
        }
        
        # Expected data schemas
        self.transaction_schema = {
            'trans_date_trans_time': 'datetime64[ns]',
            'cc_num': 'int64',
            'merchant': 'object',
            'category': 'object',
            'amt': 'float64',
            'unix_time': 'int64',
            'merch_lat': 'float64',
            'merch_long': 'float64',
            'is_fraud': 'int64'
        }
        
        # Business rules for validation
        self.business_rules = {
            'amt': {'min': 0.01, 'max': 50000.0},
            'merch_lat': {'min': -90.0, 'max': 90.0},
            'merch_long': {'min': -180.0, 'max': 180.0},
            'cc_num': {'min_length': 13, 'max_length': 19},
            'category': {'allowed_values': ['gas_transport', 'grocery_pos', 'home', 'kids_pets', 
                        'misc_pos', 'entertainment', 'food_dining', 'personal_care', 
                        'health_fitness', 'travel', 'shopping_pos', 'shopping_net']}
        }
        
    def validate_dataset(self, df: pd.DataFrame, dataset_type: str = "transaction") -> DataQualityMetrics:
        """Comprehensive dataset validation with quality scoring"""
        
        issues = []
        recommendations = []
        
        print(f"🔍 Starting comprehensive data quality validation for {len(df)} records...")
        
        # 1. Completeness Assessment
        completeness_score, comp_issues, comp_recs = self._assess_completeness(df)
        issues.extend(comp_issues)
        recommendations.extend(comp_recs)
        
        # 2. Consistency Assessment  
        consistency_score, cons_issues, cons_recs = self._assess_consistency(df)
        issues.extend(cons_issues)
        recommendations.extend(cons_recs)
        
        # 3. Validity Assessment
        validity_score, val_issues, val_recs = self._assess_validity(df)
        issues.extend(val_issues)
        recommendations.extend(val_recs)
        
        # 4. Uniqueness Assessment
        uniqueness_score, uniq_issues, uniq_recs = self._assess_uniqueness(df)
        issues.extend(uniq_issues)
        recommendations.extend(uniq_recs)
        
        # 5. Timeliness Assessment
        timeliness_score, time_issues, time_recs = self._assess_timeliness(df)
        issues.extend(time_issues)
        recommendations.extend(time_recs)
        
        # Calculate overall quality score (weighted)
        weights = {'completeness': 0.25, 'consistency': 0.20, 'validity': 0.25, 
                  'uniqueness': 0.15, 'timeliness': 0.15}
        
        overall_score = (
            completeness_score * weights['completeness'] +
            consistency_score * weights['consistency'] +
            validity_score * weights['validity'] +
            uniqueness_score * weights['uniqueness'] +
            timeliness_score * weights['timeliness']
        )
        
        print(f"✅ Data quality validation completed!")
        print(f"📊 Overall Quality Score: {overall_score:.2%}")
        
        return DataQualityMetrics(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            validity_score=validity_score,
            uniqueness_score=uniqueness_score,
            timeliness_score=timeliness_score,
            overall_score=overall_score,
            issues_found=issues,
            recommendations=recommendations
        )
    
    def _assess_completeness(self, df: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """Assess data completeness"""
        issues = []
        recommendations = []
        
        # Calculate missing data percentages
        missing_data = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        missing_percentage = missing_data.sum() / total_cells
        
        completeness_score = 1.0 - missing_percentage
        
        # Identify problematic columns
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                missing_pct = missing_count / len(df)
                if missing_pct > 0.05:  # More than 5% missing
                    issues.append(f"Column '{col}' has {missing_pct:.2%} missing values")
                    if missing_pct > 0.20:
                        recommendations.append(f"Consider removing column '{col}' or implementing advanced imputation")
                    else:
                        recommendations.append(f"Implement missing value imputation for column '{col}'")
        
        if completeness_score < self.quality_thresholds['completeness']:
            issues.append(f"Overall completeness ({completeness_score:.2%}) below threshold ({self.quality_thresholds['completeness']:.2%})")
        
        return completeness_score, issues, recommendations
    
    def _assess_consistency(self, df: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """Assess data consistency"""
        issues = []
        recommendations = []
        consistency_violations = 0
        total_checks = 0
        
        # Check datetime consistency
        if 'trans_date_trans_time' in df.columns and 'unix_time' in df.columns:
            total_checks += len(df)
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df['trans_date_trans_time']):
                    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
                
                # Check if unix_time matches trans_date_trans_time
                expected_unix = df['trans_date_trans_time'].astype('int64') // 10**9
                time_diff = abs(df['unix_time'] - expected_unix)
                inconsistent_times = (time_diff > 3600).sum()  # More than 1 hour difference
                
                if inconsistent_times > 0:
                    consistency_violations += inconsistent_times
                    issues.append(f"{inconsistent_times} records have inconsistent timestamp fields")
                    recommendations.append("Standardize timestamp generation process")
                    
            except Exception as e:
                issues.append(f"Unable to validate timestamp consistency: {str(e)}")
        
        # Check amount consistency
        if 'amt' in df.columns:
            total_checks += len(df)
            negative_amounts = (df['amt'] < 0).sum()
            zero_amounts = (df['amt'] == 0).sum()
            
            if negative_amounts > 0:
                consistency_violations += negative_amounts
                issues.append(f"{negative_amounts} records have negative transaction amounts")
                recommendations.append("Implement business rule validation for transaction amounts")
            
            if zero_amounts > len(df) * 0.01:  # More than 1% zero amounts
                issues.append(f"High number of zero-amount transactions: {zero_amounts}")
        
        # Check coordinate consistency
        if 'merch_lat' in df.columns and 'merch_long' in df.columns:
            total_checks += len(df)
            invalid_coords = (
                (df['merch_lat'].abs() > 90) | 
                (df['merch_long'].abs() > 180) |
                ((df['merch_lat'] == 0) & (df['merch_long'] == 0))
            ).sum()
            
            if invalid_coords > 0:
                consistency_violations += invalid_coords
                issues.append(f"{invalid_coords} records have invalid coordinates")
                recommendations.append("Implement coordinate validation and geocoding")
        
        consistency_score = 1.0 - (consistency_violations / max(total_checks, 1))
        
        return consistency_score, issues, recommendations
    
    def _assess_validity(self, df: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """Assess data validity against business rules"""
        issues = []
        recommendations = []
        validity_violations = 0
        total_checks = 0
        
        # Validate against business rules
        for col, rules in self.business_rules.items():
            if col in df.columns:
                total_checks += len(df)
                
                if 'min' in rules and 'max' in rules:
                    # Numeric range validation
                    invalid_range = ((df[col] < rules['min']) | (df[col] > rules['max'])).sum()
                    if invalid_range > 0:
                        validity_violations += invalid_range
                        issues.append(f"{invalid_range} records in '{col}' outside valid range [{rules['min']}, {rules['max']}]")
                
                if 'allowed_values' in rules:
                    # Categorical validation
                    invalid_categories = ~df[col].isin(rules['allowed_values'])
                    invalid_count = invalid_categories.sum()
                    if invalid_count > 0:
                        validity_violations += invalid_count
                        unique_invalid = df[invalid_categories][col].unique()
                        issues.append(f"{invalid_count} records in '{col}' have invalid categories: {list(unique_invalid)[:5]}")
                        recommendations.append(f"Standardize category values for '{col}' column")
                
                if 'min_length' in rules and 'max_length' in rules:
                    # String length validation
                    str_lengths = df[col].astype(str).str.len()
                    invalid_length = ((str_lengths < rules['min_length']) | 
                                    (str_lengths > rules['max_length'])).sum()
                    if invalid_length > 0:
                        validity_violations += invalid_length
                        issues.append(f"{invalid_length} records in '{col}' have invalid length")
        
        # Data type validation
        for col, expected_dtype in self.transaction_schema.items():
            if col in df.columns:
                total_checks += len(df)
                try:
                    if expected_dtype.startswith('datetime'):
                        if not pd.api.types.is_datetime64_any_dtype(df[col]):
                            pd.to_datetime(df[col])  # Test conversion
                    elif expected_dtype == 'int64':
                        pd.to_numeric(df[col], errors='raise')
                    elif expected_dtype == 'float64':
                        pd.to_numeric(df[col], errors='raise')
                except:
                    validity_violations += len(df)
                    issues.append(f"Column '{col}' cannot be converted to expected type '{expected_dtype}'")
                    recommendations.append(f"Implement data type conversion for '{col}'")
        
        validity_score = 1.0 - (validity_violations / max(total_checks, 1))
        
        return validity_score, issues, recommendations
    
    def _assess_uniqueness(self, df: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """Assess data uniqueness"""
        issues = []
        recommendations = []
        
        # Check for duplicate records
        total_duplicates = df.duplicated().sum()
        uniqueness_score = 1.0 - (total_duplicates / len(df))
        
        if total_duplicates > 0:
            issues.append(f"{total_duplicates} duplicate records found")
            recommendations.append("Implement deduplication process before training")
        
        # Check key field uniqueness (if applicable)
        if 'cc_num' in df.columns and 'trans_date_trans_time' in df.columns:
            # Transaction-level uniqueness
            key_duplicates = df.duplicated(subset=['cc_num', 'trans_date_trans_time']).sum()
            if key_duplicates > 0:
                issues.append(f"{key_duplicates} records have duplicate transaction keys")
        
        return uniqueness_score, issues, recommendations
    
    def _assess_timeliness(self, df: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """Assess data timeliness"""
        issues = []
        recommendations = []
        timeliness_score = 1.0
        
        if 'trans_date_trans_time' in df.columns:
            try:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(df['trans_date_trans_time']):
                    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
                
                now = datetime.now()
                future_transactions = (df['trans_date_trans_time'] > now).sum()
                
                if future_transactions > 0:
                    timeliness_score -= 0.1
                    issues.append(f"{future_transactions} transactions have future timestamps")
                    recommendations.append("Validate timestamp generation to prevent future dates")
                
                # Check data freshness
                latest_transaction = df['trans_date_trans_time'].max()
                days_old = (now - latest_transaction).days
                
                if days_old > 30:
                    timeliness_score -= 0.2
                    issues.append(f"Latest transaction is {days_old} days old")
                    recommendations.append("Update dataset with more recent transaction data")
                elif days_old > 7:
                    timeliness_score -= 0.1
                    issues.append(f"Data is {days_old} days old - consider refreshing")
                
            except Exception as e:
                timeliness_score = 0.5
                issues.append(f"Unable to assess timeliness: {str(e)}")
        
        return timeliness_score, issues, recommendations
    
    def generate_quality_report(self, metrics: DataQualityMetrics, save_path: str = None) -> str:
        """Generate comprehensive data quality report"""
        
        report = f"""
# 🔍 DATA QUALITY ASSESSMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 QUALITY SCORES SUMMARY
- **Overall Score**: {metrics.overall_score:.1%} {'✅' if metrics.overall_score > 0.85 else '⚠️' if metrics.overall_score > 0.70 else '❌'}
- **Completeness**: {metrics.completeness_score:.1%} {'✅' if metrics.completeness_score > 0.90 else '⚠️'}
- **Consistency**: {metrics.consistency_score:.1%} {'✅' if metrics.consistency_score > 0.85 else '⚠️'}
- **Validity**: {metrics.validity_score:.1%} {'✅' if metrics.validity_score > 0.80 else '⚠️'}
- **Uniqueness**: {metrics.uniqueness_score:.1%} {'✅' if metrics.uniqueness_score > 0.95 else '⚠️'}
- **Timeliness**: {metrics.timeliness_score:.1%} {'✅' if metrics.timeliness_score > 0.85 else '⚠️'}

## ⚠️ ISSUES IDENTIFIED ({len(metrics.issues_found)})
"""
        
        for i, issue in enumerate(metrics.issues_found, 1):
            report += f"{i}. {issue}\n"
        
        if not metrics.issues_found:
            report += "No significant issues detected! 🎉\n"
        
        report += f"""
## 💡 RECOMMENDATIONS ({len(metrics.recommendations)})
"""
        
        for i, rec in enumerate(metrics.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        if not metrics.recommendations:
            report += "Dataset meets all quality standards! 🌟\n"
        
        report += """
## 🎯 QUALITY ASSESSMENT LEGEND
- ✅ Excellent (>90%)
- ⚠️ Acceptable (70-90%)
- ❌ Needs Improvement (<70%)

## 📋 NEXT STEPS
1. Address critical issues (❌) first
2. Implement recommended improvements
3. Re-run validation after fixes
4. Monitor quality metrics in production
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"📄 Quality report saved to: {save_path}")
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Create sample test data
    np.random.seed(42)
    
    sample_data = {
        'trans_date_trans_time': pd.date_range('2023-01-01', periods=1000, freq='1H'),
        'cc_num': np.random.randint(4000000000000000, 5000000000000000, 1000),
        'merchant': np.random.choice(['Store_A', 'Store_B', 'Store_C', 'Online_Shop'], 1000),
        'category': np.random.choice(['grocery_pos', 'gas_transport', 'entertainment', 'food_dining'], 1000),
        'amt': np.random.lognormal(3, 1, 1000),  # Log-normal distribution for amounts
        'unix_time': [int(dt.timestamp()) for dt in pd.date_range('2023-01-01', periods=1000, freq='1H')],
        'merch_lat': np.random.uniform(-90, 90, 1000),
        'merch_long': np.random.uniform(-180, 180, 1000),
        'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    }
    
    # Add some quality issues for testing
    sample_df = pd.DataFrame(sample_data)
    
    # Introduce missing values
    sample_df.loc[10:15, 'merchant'] = None
    
    # Introduce invalid amounts
    sample_df.loc[20:25, 'amt'] = -100
    
    # Introduce future dates
    sample_df.loc[30:35, 'trans_date_trans_time'] = datetime.now() + timedelta(days=10)
    
    # Test the validator
    validator = AdvancedDataQualityValidator()
    metrics = validator.validate_dataset(sample_df)
    
    # Generate and print report
    report = validator.generate_quality_report(metrics)
    print(report)