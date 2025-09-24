"""
Automated Data Drift Detection System
Part of SecureBank Phase 4: Enhanced Dataset Generation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
import json
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DriftResult:
    """Data drift detection result"""
    feature_name: str
    drift_detected: bool
    drift_score: float
    drift_type: str  # 'distributional', 'statistical', 'categorical'
    test_statistic: float
    p_value: float
    threshold: float
    recommendation: str

@dataclass
class DriftReport:
    """Comprehensive drift report"""
    timestamp: str
    baseline_period: str
    comparison_period: str
    total_features: int
    features_with_drift: int
    drift_percentage: float
    overall_severity: str
    feature_results: List[DriftResult]
    recommendations: List[str]

class DataDriftDetector:
    """Advanced data drift detection with multiple statistical tests"""
    
    def __init__(self, drift_threshold: float = 0.05, 
                 severity_thresholds: Dict[str, float] = None):
        """
        Initialize drift detector
        
        Args:
            drift_threshold: P-value threshold for drift detection (default 0.05)
            severity_thresholds: Custom thresholds for drift severity levels
        """
        self.drift_threshold = drift_threshold
        self.severity_thresholds = severity_thresholds or {
            'low': 0.01,
            'medium': 0.001, 
            'high': 0.0001
        }
        
        # Store baseline statistics
        self.baseline_stats = {}
        self.baseline_distributions = {}
        
    def set_baseline(self, baseline_df: pd.DataFrame, 
                    feature_columns: List[str] = None) -> None:
        """
        Set baseline dataset for drift comparison
        
        Args:
            baseline_df: Baseline dataset
            feature_columns: Specific columns to monitor (if None, uses all numeric columns)
        """
        
        print(f"📊 Setting baseline with {len(baseline_df)} records...")
        
        if feature_columns is None:
            # Use all numeric columns and categorical with reasonable cardinality
            numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = []
            
            for col in baseline_df.select_dtypes(include=['object']).columns:
                if baseline_df[col].nunique() <= 50:  # Reasonable cardinality
                    categorical_cols.append(col)
            
            feature_columns = numeric_cols + categorical_cols
        
        self.monitored_features = feature_columns
        
        # Calculate baseline statistics
        for col in feature_columns:
            if col not in baseline_df.columns:
                continue
                
            self.baseline_stats[col] = {}
            
            if baseline_df[col].dtype in ['object', 'category']:
                # Categorical feature
                self.baseline_stats[col]['type'] = 'categorical'
                self.baseline_stats[col]['value_counts'] = baseline_df[col].value_counts(normalize=True)
                self.baseline_stats[col]['unique_values'] = set(baseline_df[col].unique())
                
            else:
                # Numeric feature
                self.baseline_stats[col]['type'] = 'numeric'
                self.baseline_stats[col]['mean'] = baseline_df[col].mean()
                self.baseline_stats[col]['std'] = baseline_df[col].std()
                self.baseline_stats[col]['median'] = baseline_df[col].median()
                self.baseline_stats[col]['q25'] = baseline_df[col].quantile(0.25)
                self.baseline_stats[col]['q75'] = baseline_df[col].quantile(0.75)
                self.baseline_stats[col]['min'] = baseline_df[col].min()
                self.baseline_stats[col]['max'] = baseline_df[col].max()
                
                # Store distribution for KS test
                self.baseline_distributions[col] = baseline_df[col].dropna().values
        
        print(f"✅ Baseline set for {len(self.monitored_features)} features")
        
    def detect_drift(self, comparison_df: pd.DataFrame, 
                    detailed_analysis: bool = True) -> DriftReport:
        """
        Detect data drift between baseline and comparison datasets
        
        Args:
            comparison_df: Dataset to compare against baseline
            detailed_analysis: Whether to perform detailed statistical analysis
        
        Returns:
            DriftReport with comprehensive drift analysis
        """
        
        if not self.baseline_stats:
            raise ValueError("Baseline not set. Call set_baseline() first.")
        
        print(f"🔍 Detecting drift in {len(comparison_df)} records...")
        
        drift_results = []
        features_with_drift = 0
        
        for feature in self.monitored_features:
            if feature not in comparison_df.columns:
                continue
                
            result = self._test_feature_drift(feature, comparison_df[feature], detailed_analysis)
            drift_results.append(result)
            
            if result.drift_detected:
                features_with_drift += 1
        
        # Calculate overall metrics
        drift_percentage = (features_with_drift / len(drift_results)) * 100 if drift_results else 0
        overall_severity = self._determine_overall_severity(drift_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(drift_results, drift_percentage)
        
        report = DriftReport(
            timestamp=datetime.now().isoformat(),
            baseline_period="baseline_data",
            comparison_period="current_data", 
            total_features=len(drift_results),
            features_with_drift=features_with_drift,
            drift_percentage=drift_percentage,
            overall_severity=overall_severity,
            feature_results=drift_results,
            recommendations=recommendations
        )
        
        print(f"📈 Drift detection completed: {features_with_drift}/{len(drift_results)} features showing drift")
        
        return report
    
    def _test_feature_drift(self, feature_name: str, comparison_data: pd.Series, 
                           detailed: bool = True) -> DriftResult:
        """Test for drift in a specific feature"""
        
        baseline_info = self.baseline_stats[feature_name]
        comparison_clean = comparison_data.dropna()
        
        if baseline_info['type'] == 'categorical':
            return self._test_categorical_drift(feature_name, comparison_clean, baseline_info)
        else:
            return self._test_numeric_drift(feature_name, comparison_clean, baseline_info, detailed)
    
    def _test_categorical_drift(self, feature_name: str, comparison_data: pd.Series, 
                              baseline_info: Dict) -> DriftResult:
        """Test drift for categorical features using Chi-square test"""
        
        baseline_counts = baseline_info['value_counts']
        comparison_counts = comparison_data.value_counts(normalize=True)
        
        # Align categories
        all_categories = set(baseline_counts.index) | set(comparison_counts.index)
        
        baseline_aligned = []
        comparison_aligned = []
        
        for cat in all_categories:
            baseline_aligned.append(baseline_counts.get(cat, 0))
            comparison_aligned.append(comparison_counts.get(cat, 0))
        
        # Chi-square test
        try:
            # Convert proportions to expected counts for chi-square
            baseline_expected = np.array(baseline_aligned) * len(comparison_data)
            comparison_observed = np.array(comparison_aligned) * len(comparison_data)
            
            # Add small constant to avoid zero expected frequencies
            baseline_expected = baseline_expected + 1e-8
            comparison_observed = comparison_observed + 1e-8
            
            chi2_stat, p_value = stats.chisquare(comparison_observed, baseline_expected)
            
            drift_detected = p_value < self.drift_threshold
            drift_score = 1 - p_value  # Higher score = more drift
            
            # Check for new categories
            new_categories = set(comparison_counts.index) - baseline_info['unique_values']
            if new_categories:
                recommendation = f"New categories detected: {list(new_categories)[:5]}. Consider retraining."
            else:
                recommendation = "Monitor category distribution changes" if drift_detected else "No action needed"
                
        except Exception as e:
            chi2_stat, p_value = 0, 1
            drift_detected = False
            drift_score = 0
            recommendation = f"Unable to perform chi-square test: {str(e)}"
        
        return DriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=drift_score,
            drift_type='categorical',
            test_statistic=chi2_stat,
            p_value=p_value,
            threshold=self.drift_threshold,
            recommendation=recommendation
        )
    
    def _test_numeric_drift(self, feature_name: str, comparison_data: pd.Series, 
                           baseline_info: Dict, detailed: bool = True) -> DriftResult:
        """Test drift for numeric features using multiple tests"""
        
        baseline_data = self.baseline_distributions[feature_name]
        
        # Primary test: Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.ks_2samp(baseline_data, comparison_data.values)
        
        drift_detected = ks_p_value < self.drift_threshold
        drift_score = ks_stat  # KS statistic as drift score
        
        # Additional tests if detailed analysis requested
        additional_info = ""
        if detailed:
            # Mann-Whitney U test for distribution shift
            try:
                mw_stat, mw_p_value = stats.mannwhitneyu(baseline_data, comparison_data.values, 
                                                        alternative='two-sided')
                if mw_p_value < self.drift_threshold:
                    additional_info += f"Distribution shift detected (MW p={mw_p_value:.4f}). "
            except:
                pass
            
            # Check for mean shift
            baseline_mean = baseline_info['mean']
            comparison_mean = comparison_data.mean()
            mean_shift = abs(comparison_mean - baseline_mean) / (baseline_info['std'] + 1e-8)
            
            if mean_shift > 2:  # More than 2 standard deviations
                additional_info += f"Significant mean shift: {mean_shift:.2f} std devs. "
            
            # Check for variance change
            baseline_std = baseline_info['std']
            comparison_std = comparison_data.std()
            variance_ratio = comparison_std / (baseline_std + 1e-8)
            
            if variance_ratio > 2 or variance_ratio < 0.5:
                additional_info += f"Variance change detected: {variance_ratio:.2f}x baseline. "
        
        # Generate recommendation
        if drift_detected:
            if ks_stat > 0.3:
                recommendation = f"High drift detected. {additional_info}Consider immediate model retraining."
            elif ks_stat > 0.15:
                recommendation = f"Moderate drift detected. {additional_info}Schedule model validation."
            else:
                recommendation = f"Low drift detected. {additional_info}Monitor closely."
        else:
            recommendation = "No significant drift detected."
        
        return DriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=drift_score,
            drift_type='distributional',
            test_statistic=ks_stat,
            p_value=ks_p_value,
            threshold=self.drift_threshold,
            recommendation=recommendation
        )
    
    def _determine_overall_severity(self, results: List[DriftResult]) -> str:
        """Determine overall drift severity"""
        
        if not results:
            return "unknown"
        
        drift_results = [r for r in results if r.drift_detected]
        
        if not drift_results:
            return "none"
        
        drift_percentage = len(drift_results) / len(results)
        avg_drift_score = np.mean([r.drift_score for r in drift_results])
        
        if drift_percentage > 0.5 or avg_drift_score > 0.3:
            return "high"
        elif drift_percentage > 0.25 or avg_drift_score > 0.15:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, results: List[DriftResult], 
                                drift_percentage: float) -> List[str]:
        """Generate actionable recommendations based on drift analysis"""
        
        recommendations = []
        
        high_drift_features = [r for r in results if r.drift_detected and r.drift_score > 0.3]
        medium_drift_features = [r for r in results if r.drift_detected and 0.15 < r.drift_score <= 0.3]
        
        if drift_percentage > 50:
            recommendations.append("🚨 CRITICAL: Over 50% of features show drift. Immediate model retraining required.")
        elif drift_percentage > 25:
            recommendations.append("⚠️ HIGH: Significant drift detected. Schedule model validation and potential retraining.")
        elif drift_percentage > 10:
            recommendations.append("📊 MEDIUM: Some drift detected. Increase monitoring frequency.")
        
        if high_drift_features:
            feature_names = [r.feature_name for r in high_drift_features[:3]]
            recommendations.append(f"🎯 Priority features for immediate attention: {', '.join(feature_names)}")
        
        if medium_drift_features:
            recommendations.append(f"📈 Monitor closely: {len(medium_drift_features)} features showing moderate drift")
        
        # Feature-specific recommendations
        categorical_drift = [r for r in results if r.drift_type == 'categorical' and r.drift_detected]
        if categorical_drift:
            recommendations.append("🏷️ Categorical drift detected - check for new data sources or encoding changes")
        
        distributional_drift = [r for r in results if r.drift_type == 'distributional' and r.drift_detected]
        if distributional_drift:
            recommendations.append("📊 Distributional drift detected - validate data processing pipeline")
        
        if not any(r.drift_detected for r in results):
            recommendations.append("✅ No significant drift detected. Continue normal monitoring.")
        
        return recommendations
    
    def monitor_drift_over_time(self, data_batches: List[Tuple[str, pd.DataFrame]], 
                               save_results: bool = True) -> List[DriftReport]:
        """
        Monitor drift across multiple time periods
        
        Args:
            data_batches: List of (period_name, dataframe) tuples
            save_results: Whether to save results to files
        
        Returns:
            List of drift reports for each period
        """
        
        print(f"📊 Monitoring drift across {len(data_batches)} time periods...")
        
        reports = []
        
        for period_name, batch_df in data_batches:
            print(f"\n🔍 Analyzing period: {period_name}")
            
            report = self.detect_drift(batch_df)
            report.comparison_period = period_name
            reports.append(report)
            
            if save_results:
                # Save individual report
                filename = f"drift_report_{period_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.save_drift_report(report, filename)
        
        # Generate trend analysis
        if len(reports) > 1:
            trend_analysis = self._analyze_drift_trends(reports)
            print(f"\n📈 Drift Trend Analysis:")
            print(f"   Average drift percentage: {trend_analysis['avg_drift_percentage']:.1f}%")
            print(f"   Trend: {trend_analysis['trend']}")
            print(f"   Most problematic features: {', '.join(trend_analysis['top_drift_features'][:3])}")
        
        return reports
    
    def _analyze_drift_trends(self, reports: List[DriftReport]) -> Dict[str, Any]:
        """Analyze trends across multiple drift reports"""
        
        drift_percentages = [r.drift_percentage for r in reports]
        feature_drift_counts = {}
        
        # Count how often each feature shows drift
        for report in reports:
            for result in report.feature_results:
                if result.drift_detected:
                    feature_drift_counts[result.feature_name] = feature_drift_counts.get(result.feature_name, 0) + 1
        
        # Determine trend
        if len(drift_percentages) >= 3:
            recent_avg = np.mean(drift_percentages[-3:])
            earlier_avg = np.mean(drift_percentages[:-3]) if len(drift_percentages) > 3 else drift_percentages[0]
            
            if recent_avg > earlier_avg * 1.2:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.8:
                trend = "decreasing" 
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        # Most problematic features
        top_drift_features = sorted(feature_drift_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'avg_drift_percentage': np.mean(drift_percentages),
            'trend': trend,
            'top_drift_features': [f[0] for f in top_drift_features],
            'feature_drift_frequency': dict(top_drift_features)
        }
    
    def save_drift_report(self, report: DriftReport, filename: str = None) -> str:
        """Save drift report to JSON file"""
        
        if filename is None:
            filename = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert dataclass to dict for JSON serialization
        report_dict = asdict(report)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Apply conversion to the entire report
        report_dict = convert_numpy_types(report_dict)
        
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"📄 Drift report saved to: {filename}")
        return filename
    
    def generate_drift_summary_report(self, report: DriftReport, save_path: str = None) -> str:
        """Generate human-readable drift summary report"""
        
        severity_emoji = {
            'none': '✅',
            'low': '🟡', 
            'medium': '🟠',
            'high': '🔴',
            'unknown': '❓'
        }
        
        summary = f"""
# 🔍 DATA DRIFT DETECTION REPORT
Generated: {report.timestamp}

## 📊 EXECUTIVE SUMMARY
- **Overall Severity**: {severity_emoji.get(report.overall_severity, '❓')} {report.overall_severity.upper()}
- **Features Analyzed**: {report.total_features}
- **Features with Drift**: {report.features_with_drift}
- **Drift Percentage**: {report.drift_percentage:.1f}%

## 🚨 DRIFT ALERTS
"""
        
        # Categorize features by drift severity
        high_drift = [r for r in report.feature_results if r.drift_detected and r.drift_score > 0.3]
        medium_drift = [r for r in report.feature_results if r.drift_detected and 0.15 < r.drift_score <= 0.3]
        low_drift = [r for r in report.feature_results if r.drift_detected and r.drift_score <= 0.15]
        
        if high_drift:
            summary += f"\n### 🔴 HIGH PRIORITY ({len(high_drift)} features)\n"
            for result in high_drift:
                summary += f"- **{result.feature_name}**: Score {result.drift_score:.3f}, p-value {result.p_value:.4f}\n"
                summary += f"  💡 {result.recommendation}\n"
        
        if medium_drift:
            summary += f"\n### 🟠 MEDIUM PRIORITY ({len(medium_drift)} features)\n"
            for result in medium_drift:
                summary += f"- **{result.feature_name}**: Score {result.drift_score:.3f}, p-value {result.p_value:.4f}\n"
        
        if low_drift:
            summary += f"\n### 🟡 LOW PRIORITY ({len(low_drift)} features)\n"
            for result in low_drift:
                summary += f"- {result.feature_name}: Score {result.drift_score:.3f}\n"
        
        summary += f"""
## 💡 RECOMMENDATIONS
"""
        for i, rec in enumerate(report.recommendations, 1):
            summary += f"{i}. {rec}\n"
        
        summary += f"""
## 📈 DETAILED ANALYSIS

### Drift Detection Statistics
- **Detection Threshold**: {report.feature_results[0].threshold if report.feature_results else 'N/A'}
- **Statistical Tests Used**: 
  - Kolmogorov-Smirnov (numeric features)
  - Chi-square (categorical features)
  - Mann-Whitney U (distribution shift)

### Feature Breakdown by Type
"""
        
        categorical_features = [r for r in report.feature_results if r.drift_type == 'categorical']
        distributional_features = [r for r in report.feature_results if r.drift_type == 'distributional']
        
        summary += f"- **Categorical Features**: {len(categorical_features)} ({len([r for r in categorical_features if r.drift_detected])} with drift)\n"
        summary += f"- **Numeric Features**: {len(distributional_features)} ({len([r for r in distributional_features if r.drift_detected])} with drift)\n"
        
        summary += f"""
## 🔧 TECHNICAL DETAILS

### Most Significant Drifts
"""
        
        # Sort by drift score for technical details
        sorted_results = sorted([r for r in report.feature_results if r.drift_detected], 
                              key=lambda x: x.drift_score, reverse=True)
        
        for result in sorted_results[:5]:  # Top 5 most significant
            summary += f"""
**{result.feature_name}**
- Type: {result.drift_type}
- Drift Score: {result.drift_score:.4f}
- Test Statistic: {result.test_statistic:.4f}
- P-value: {result.p_value:.6f}
- Recommendation: {result.recommendation}
"""
        
        summary += f"""
## 📅 MONITORING SCHEDULE
Based on drift severity, recommended monitoring frequency:
- **High Drift Features**: Daily monitoring
- **Medium Drift Features**: Weekly monitoring  
- **Low Drift Features**: Monthly monitoring
- **Stable Features**: Quarterly monitoring

## ⚠️ ACTION ITEMS
1. **Immediate** ({len(high_drift)} features): Investigate root causes and consider model retraining
2. **Short-term** ({len(medium_drift)} features): Increase monitoring and prepare for potential intervention
3. **Long-term** ({len(low_drift)} features): Monitor trends and document changes

---
*This report was generated by SecureBank's automated drift detection system.*
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(summary)
            print(f"📄 Drift summary report saved to: {save_path}")
        
        return summary
    
    def setup_automated_monitoring(self, monitoring_schedule: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Setup automated drift monitoring configuration
        
        Args:
            monitoring_schedule: Custom schedule in hours for different drift levels
        
        Returns:
            Configuration dictionary for automated monitoring
        """
        
        default_schedule = {
            'high_drift': 24,      # Every 24 hours
            'medium_drift': 168,   # Weekly (168 hours)
            'low_drift': 720,      # Monthly (720 hours)
            'stable': 2160         # Quarterly (2160 hours)
        }
        
        schedule = monitoring_schedule or default_schedule
        
        config = {
            'monitoring_enabled': True,
            'baseline_update_frequency': 2160,  # Update baseline quarterly
            'alert_thresholds': {
                'drift_percentage': 25,  # Alert if >25% features drift
                'high_drift_count': 3,   # Alert if >3 features have high drift
                'severity_level': 'medium'  # Alert on medium+ severity
            },
            'schedule': schedule,
            'notification_settings': {
                'email_alerts': True,
                'dashboard_alerts': True,
                'log_alerts': True
            },
            'auto_actions': {
                'retrain_threshold': 0.5,  # Auto-retrain if >50% features drift
                'validation_threshold': 0.25,  # Auto-validate if >25% features drift
                'backup_model': True  # Keep backup of previous model
            }
        }
        
        return config

# Integration functions for SecureBank system
class SecureBankDriftMonitor:
    """Integration class for SecureBank drift monitoring"""
    
    def __init__(self, data_path: str = "securebank/storage/datasets"):
        self.data_path = data_path
        self.detector = DataDriftDetector()
        self.monitoring_config = None
        
    def initialize_monitoring(self, baseline_dataset_path: str) -> bool:
        """Initialize drift monitoring with baseline dataset"""
        
        try:
            # Load baseline data
            baseline_df = pd.read_csv(baseline_dataset_path)
            print(f"📊 Loading baseline from: {baseline_dataset_path}")
            
            # Set baseline for drift detection
            self.detector.set_baseline(baseline_df)
            
            # Setup automated monitoring
            self.monitoring_config = self.detector.setup_automated_monitoring()
            
            print("✅ Drift monitoring initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize drift monitoring: {str(e)}")
            return False
    
    def check_new_data_drift(self, new_dataset_path: str) -> DriftReport:
        """Check drift in newly created dataset"""
        
        try:
            # Load new data
            new_df = pd.read_csv(new_dataset_path)
            print(f"🔍 Checking drift in: {new_dataset_path}")
            
            # Detect drift
            report = self.detector.detect_drift(new_df)
            
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = f"securebank/logs/drift_report_{timestamp}.json"
            self.detector.save_drift_report(report, report_path)
            
            # Generate summary
            summary_path = f"securebank/logs/drift_summary_{timestamp}.txt"
            self.detector.generate_drift_summary_report(report, summary_path)
            
            return report
            
        except Exception as e:
            print(f"❌ Failed to check drift: {str(e)}")
            raise
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        
        return {
            'monitoring_initialized': bool(self.detector.baseline_stats),
            'monitored_features': len(self.detector.monitored_features) if hasattr(self.detector, 'monitored_features') else 0,
            'baseline_set': bool(self.detector.baseline_stats),
            'config': self.monitoring_config
        }

# Example usage and testing
if __name__ == "__main__":
    # Create sample datasets for testing
    np.random.seed(42)
    
    # Baseline dataset
    baseline_data = {
        'trans_date_trans_time': pd.date_range('2023-01-01', periods=1000, freq='1H'),
        'cc_num': np.random.choice(range(1000, 2000), 1000),
        'merchant': np.random.choice(['Store_A', 'Store_B', 'Store_C'], 1000),
        'category': np.random.choice(['grocery_pos', 'gas_transport', 'entertainment'], 1000),
        'amt': np.random.normal(50, 20, 1000),
        'merch_lat': np.random.normal(40, 5, 1000),
        'merch_long': np.random.normal(-100, 10, 1000),
        'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    }
    baseline_df = pd.DataFrame(baseline_data)
    
    # Comparison dataset with introduced drift
    comparison_data = baseline_data.copy()
    
    # Introduce drift in amounts (mean shift)
    comparison_data['amt'] = np.random.normal(75, 25, 1000)  # Higher mean and variance
    
    # Introduce drift in categories (new category)
    comparison_data['category'] = np.random.choice(['grocery_pos', 'gas_transport', 'entertainment', 'online_shopping'], 1000)
    
    # Introduce drift in locations (shift distribution)
    comparison_data['merch_lat'] = np.random.normal(35, 5, 1000)  # Shifted south
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Test drift detection
    detector = DataDriftDetector(drift_threshold=0.05)
    
    print("Setting baseline...")
    detector.set_baseline(baseline_df)
    
    print("\nDetecting drift...")
    report = detector.detect_drift(comparison_df)
    
    print(f"\nDrift Detection Results:")
    print(f"Features with drift: {report.features_with_drift}/{report.total_features}")
    print(f"Drift percentage: {report.drift_percentage:.1f}%")
    print(f"Overall severity: {report.overall_severity}")
    
    # Generate and display summary report
    summary = detector.generate_drift_summary_report(report)
    print("\n" + "="*60)
    print(summary[:1000] + "..." if len(summary) > 1000 else summary)