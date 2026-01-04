# securebank/modules/data/drift_detector.py
"""
Data drift detection system for SecureBank fraud detection datasets.
Monitors changes in data distribution and feature importance over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import json
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings


class DataDriftDetector:
    """
    Comprehensive data drift detection using statistical tests and ML-based methods.
    """
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """
        Initialize the drift detector.
        
        Parameters
        ----------
        reference_data : pd.DataFrame, optional
            Reference dataset to compare against.
        """
        self.reference_data = reference_data
        self.drift_history = []
        self.statistical_tests = {
            'numerical': ['ks_test', 'mann_whitney', 'kl_divergence'],
            'categorical': ['chi_square', 'cramers_v']
        }
        self.drift_thresholds = {
            'p_value_threshold': 0.05,
            'effect_size_threshold': 0.2,
            'kl_divergence_threshold': 0.1
        }
    
    def detect_drift(self, current_data: pd.DataFrame,
                    reference_data: Optional[pd.DataFrame] = None,
                    target_column: str = 'is_fraud') -> Dict[str, Any]:
        """
        Detect data drift between reference and current datasets.
        
        Parameters
        ----------
        current_data : pd.DataFrame
            Current dataset to analyze.
        reference_data : pd.DataFrame, optional
            Reference dataset. Uses stored reference if not provided.
        target_column : str
            Target column name.
            
        Returns
        -------
        dict
            Comprehensive drift analysis results.
        """
        if reference_data is None:
            reference_data = self.reference_data
        
        if reference_data is None:
            raise ValueError("No reference data provided")
        
        drift_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_comparison': {
                'reference_size': len(reference_data),
                'current_size': len(current_data),
                'reference_columns': list(reference_data.columns),
                'current_columns': list(current_data.columns)
            },
            'feature_drift': {},
            'target_drift': {},
            'overall_drift_score': 0.0,
            'drift_detected': False,
            'recommendations': []
        }
        
        # Align columns between datasets
        common_columns = list(set(reference_data.columns) & set(current_data.columns))
        if target_column in common_columns:
            common_columns.remove(target_column)
        
        drift_results['dataset_comparison']['common_columns'] = common_columns
        drift_results['dataset_comparison']['missing_in_current'] = list(
            set(reference_data.columns) - set(current_data.columns)
        )
        drift_results['dataset_comparison']['new_in_current'] = list(
            set(current_data.columns) - set(reference_data.columns)
        )
        
        # Analyze feature drift
        for column in common_columns:
            column_drift = self._analyze_column_drift(
                reference_data[column], 
                current_data[column], 
                column
            )
            drift_results['feature_drift'][column] = column_drift
        
        # Analyze target drift if available
        if target_column in common_columns or target_column in current_data.columns:
            if target_column in reference_data.columns:
                target_drift = self._analyze_column_drift(
                    reference_data[target_column],
                    current_data[target_column],
                    target_column
                )
                drift_results['target_drift'] = target_drift
        
        # Calculate overall drift score and detection
        drift_results['overall_drift_score'] = self._calculate_overall_drift_score(
            drift_results['feature_drift']
        )
        
        drift_results['drift_detected'] = (
            drift_results['overall_drift_score'] > self.drift_thresholds['effect_size_threshold']
        )
        
        # Generate recommendations
        drift_results['recommendations'] = self._generate_drift_recommendations(drift_results)
        
        # Store in history
        self.drift_history.append(drift_results)
        
        return drift_results
    
    def _analyze_column_drift(self, reference_series: pd.Series,
                             current_series: pd.Series,
                             column_name: str) -> Dict[str, Any]:
        """
        Analyze drift for a single column.
        
        Parameters
        ----------
        reference_series : pd.Series
            Reference data for the column.
        current_series : pd.Series
            Current data for the column.
        column_name : str
            Name of the column.
            
        Returns
        -------
        dict
            Column drift analysis results.
        """
        column_drift = {
            'column_name': column_name,
            'data_type': str(current_series.dtype),
            'statistical_tests': {},
            'distribution_changes': {},
            'drift_magnitude': 0.0,
            'drift_detected': False
        }
        
        # Remove missing values for analysis
        ref_clean = reference_series.dropna()
        curr_clean = current_series.dropna()
        
        if len(ref_clean) == 0 or len(curr_clean) == 0:
            column_drift['error'] = 'Insufficient data after removing missing values'
            return column_drift
        
        # Check if column is numerical or categorical
        is_numerical = pd.api.types.is_numeric_dtype(current_series)
        
        if is_numerical:
            column_drift.update(self._analyze_numerical_drift(ref_clean, curr_clean))
        else:
            column_drift.update(self._analyze_categorical_drift(ref_clean, curr_clean))
        
        # Calculate overall drift magnitude for this column
        test_results = column_drift['statistical_tests']
        if test_results:
            # Use the most significant test result
            p_values = [result.get('p_value', 1.0) for result in test_results.values()]
            min_p_value = min(p_values) if p_values else 1.0
            column_drift['drift_magnitude'] = 1 - min_p_value
            column_drift['drift_detected'] = min_p_value < self.drift_thresholds['p_value_threshold']
        
        return column_drift
    
    def _analyze_numerical_drift(self, ref_data: pd.Series,
                                curr_data: pd.Series) -> Dict[str, Any]:
        """Analyze drift for numerical columns."""
        results = {
            'statistical_tests': {},
            'distribution_changes': {}
        }
        
        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_p = stats.ks_2samp(ref_data, curr_data)
            results['statistical_tests']['ks_test'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_p),
                'interpretation': 'Distributions differ significantly' if ks_p < 0.05 else 'No significant difference'
            }
        except Exception as e:
            results['statistical_tests']['ks_test'] = {'error': str(e)}
        
        # Mann-Whitney U test
        try:
            mw_stat, mw_p = stats.mannwhitneyu(ref_data, curr_data, alternative='two-sided')
            results['statistical_tests']['mann_whitney'] = {
                'statistic': float(mw_stat),
                'p_value': float(mw_p),
                'interpretation': 'Medians differ significantly' if mw_p < 0.05 else 'No significant difference'
            }
        except Exception as e:
            results['statistical_tests']['mann_whitney'] = {'error': str(e)}
        
        # KL Divergence (requires binning for continuous data)
        try:
            kl_div = self._calculate_kl_divergence_continuous(ref_data, curr_data)
            results['statistical_tests']['kl_divergence'] = {
                'statistic': float(kl_div),
                'interpretation': 'High divergence' if kl_div > self.drift_thresholds['kl_divergence_threshold'] else 'Low divergence'
            }
        except Exception as e:
            results['statistical_tests']['kl_divergence'] = {'error': str(e)}
        
        # Distribution statistics comparison
        results['distribution_changes'] = {
            'mean_change': float(curr_data.mean() - ref_data.mean()),
            'std_change': float(curr_data.std() - ref_data.std()),
            'median_change': float(curr_data.median() - ref_data.median()),
            'skewness_change': float(stats.skew(curr_data) - stats.skew(ref_data)),
            'kurtosis_change': float(stats.kurtosis(curr_data) - stats.kurtosis(ref_data))
        }
        
        return results
    
    def _analyze_categorical_drift(self, ref_data: pd.Series,
                                  curr_data: pd.Series) -> Dict[str, Any]:
        """Analyze drift for categorical columns."""
        results = {
            'statistical_tests': {},
            'distribution_changes': {}
        }
        
        # Get value counts and align categories
        ref_counts = ref_data.value_counts()
        curr_counts = curr_data.value_counts()
        
        all_categories = list(set(ref_counts.index) | set(curr_counts.index))
        ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
        curr_aligned = curr_counts.reindex(all_categories, fill_value=0)
        
        # Chi-square test
        try:
            # Create contingency table
            contingency = np.array([ref_aligned.values, curr_aligned.values])
            chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)
            
            results['statistical_tests']['chi_square'] = {
                'statistic': float(chi2_stat),
                'p_value': float(chi2_p),
                'interpretation': 'Distributions differ significantly' if chi2_p < 0.05 else 'No significant difference'
            }
        except Exception as e:
            results['statistical_tests']['chi_square'] = {'error': str(e)}
        
        # Cramér's V (effect size for categorical variables)
        try:
            cramers_v = self._calculate_cramers_v(contingency)
            results['statistical_tests']['cramers_v'] = {
                'statistic': float(cramers_v),
                'interpretation': self._interpret_cramers_v(cramers_v)
            }
        except Exception as e:
            results['statistical_tests']['cramers_v'] = {'error': str(e)}
        
        # Category distribution changes
        ref_props = ref_aligned / ref_aligned.sum()
        curr_props = curr_aligned / curr_aligned.sum()
        prop_changes = curr_props - ref_props
        
        results['distribution_changes'] = {
            'new_categories': list(set(curr_counts.index) - set(ref_counts.index)),
            'missing_categories': list(set(ref_counts.index) - set(curr_counts.index)),
            'proportion_changes': prop_changes.to_dict(),
            'largest_increase': {
                'category': prop_changes.idxmax(),
                'change': float(prop_changes.max())
            },
            'largest_decrease': {
                'category': prop_changes.idxmin(),
                'change': float(prop_changes.min())
            }
        }
        
        return results
    
    def _calculate_kl_divergence_continuous(self, ref_data: pd.Series,
                                          curr_data: pd.Series,
                                          bins: int = 20) -> float:
        """Calculate KL divergence for continuous data using binning."""
        # Create bins based on the combined range
        combined_data = pd.concat([ref_data, curr_data])
        bin_edges = np.linspace(combined_data.min(), combined_data.max(), bins + 1)
        
        # Calculate histograms
        ref_hist, _ = np.histogram(ref_data, bins=bin_edges, density=True)
        curr_hist, _ = np.histogram(curr_data, bins=bin_edges, density=True)
        
        # Normalize to probabilities
        ref_prob = ref_hist / ref_hist.sum()
        curr_prob = curr_hist / curr_hist.sum()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_prob = ref_prob + epsilon
        curr_prob = curr_prob + epsilon
        
        # Calculate KL divergence
        kl_div = stats.entropy(curr_prob, ref_prob)
        return kl_div
    
    def _calculate_cramers_v(self, contingency_table: np.ndarray) -> float:
        """Calculate Cramér's V statistic."""
        chi2 = stats.chi2_contingency(contingency_table)[0]
        n = contingency_table.sum()
        r, k = contingency_table.shape
        
        cramers_v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
        return cramers_v
    
    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Interpret Cramér's V effect size."""
        if cramers_v < 0.1:
            return "Negligible association"
        elif cramers_v < 0.3:
            return "Weak association"
        elif cramers_v < 0.5:
            return "Moderate association"
        else:
            return "Strong association"
    
    def _calculate_overall_drift_score(self, feature_drift: Dict[str, Any]) -> float:
        """Calculate overall drift score across all features."""
        if not feature_drift:
            return 0.0
        
        drift_magnitudes = []
        for column_results in feature_drift.values():
            if 'drift_magnitude' in column_results:
                drift_magnitudes.append(column_results['drift_magnitude'])
        
        if not drift_magnitudes:
            return 0.0
        
        # Use mean drift magnitude across features
        return np.mean(drift_magnitudes)
    
    def _generate_drift_recommendations(self, drift_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on drift analysis."""
        recommendations = []
        
        # Overall drift recommendations
        if drift_results['drift_detected']:
            recommendations.append("Significant data drift detected - consider retraining the model")
        
        # Feature-specific recommendations
        high_drift_features = []
        for feature, results in drift_results['feature_drift'].items():
            if results.get('drift_detected', False):
                high_drift_features.append(feature)
        
        if high_drift_features:
            recommendations.append(f"High drift detected in features: {', '.join(high_drift_features)}")
            recommendations.append("Consider feature-specific preprocessing adjustments")
        
        # Target drift recommendations
        if drift_results.get('target_drift', {}).get('drift_detected', False):
            recommendations.append("Target distribution has changed - review labeling process")
        
        # Data quality recommendations
        missing_cols = drift_results['dataset_comparison'].get('missing_in_current', [])
        if missing_cols:
            recommendations.append(f"Missing columns in current data: {', '.join(missing_cols)}")
        
        new_cols = drift_results['dataset_comparison'].get('new_in_current', [])
        if new_cols:
            recommendations.append(f"New columns in current data: {', '.join(new_cols)} - consider feature importance")
        
        return recommendations
    
    def get_drift_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical drift analysis results.
        
        Parameters
        ----------
        last_n : int, optional
            Number of recent analyses to return.
            
        Returns
        -------
        list
            Historical drift analysis results.
        """
        if last_n is None:
            return self.drift_history
        else:
            return self.drift_history[-last_n:]
    
    def set_reference_data(self, reference_data: pd.DataFrame) -> None:
        """
        Set new reference data for drift detection.
        
        Parameters
        ----------
        reference_data : pd.DataFrame
            New reference dataset.
        """
        self.reference_data = reference_data
        
        # Clear history when reference data changes
        self.drift_history = []
    
    def save_drift_analysis(self, filepath: str, analysis_results: Dict[str, Any]) -> None:
        """
        Save drift analysis results to file.
        
        Parameters
        ----------
        filepath : str
            Path to save the analysis.
        analysis_results : dict
            Drift analysis results to save.
        """
        with open(filepath, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
    
    def load_drift_analysis(self, filepath: str) -> Dict[str, Any]:
        """
        Load drift analysis results from file.
        
        Parameters
        ----------
        filepath : str
            Path to load the analysis from.
            
        Returns
        -------
        dict
            Loaded drift analysis results.
        """
        with open(filepath, 'r') as f:
            return json.load(f)