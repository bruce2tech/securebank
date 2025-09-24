# securebank/modules/data/raw_data_handler.py
"""
Fixed Raw data handler that correctly handles NaN values in fraud JSON.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class RawDataHandler:
    """
    Handles loading and initial processing of raw data files.
    Fixed to handle NaN/null values in fraud_release.json
    """
    
    def __init__(self, data_dir: str = "data_sources"):
        """
        Initialize the data handler with correct paths.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = Path(data_dir)
        
        # Define file paths based on actual structure
        self.transaction_file = self.data_dir / "transactions_release.parquet"
        self.customer_file = self.data_dir / "customer_release.csv"
        self.fraud_file = self.data_dir / "fraud_release.json"
        
        # Validate that files exist
        self._validate_files()
        
    def _validate_files(self):
        """Check if all required data files exist."""
        files_status = {
            "transactions": self.transaction_file.exists(),
            "customers": self.customer_file.exists(),
            "fraud_labels": self.fraud_file.exists()
        }
        
        if not all(files_status.values()):
            missing = [k for k, v in files_status.items() if not v]
            logger.warning(f"Missing data files: {missing}")
            logger.info(f"Looking in: {self.data_dir}")
        else:
            logger.info("All data files found successfully")
            
    def load_transaction_data(self) -> pd.DataFrame:
        """
        Load transaction data from parquet file.
        
        Returns:
            DataFrame with transaction data
        """
        try:
            logger.info(f"Loading transactions from {self.transaction_file}")
            
            # Load parquet file
            transactions = pd.read_parquet(self.transaction_file)
            
            logger.info(f"Loaded {len(transactions)} transactions")
            logger.info(f"Transaction columns: {list(transactions.columns)}")
            
            # The parquet file has trans_num as index, let's preserve it
            if transactions.index.name == 'trans_num':
                # Reset index to make trans_num a regular column
                transactions = transactions.reset_index()
                logger.info("Converted trans_num from index to column")
            elif transactions.index.dtype == 'object' and 'trans_num' not in transactions.columns:
                # Index might be transaction IDs
                transactions['trans_num'] = transactions.index
            
            # Ensure required columns exist
            if 'unix_time' not in transactions.columns and 'trans_date_trans_time' in transactions.columns:
                try:
                    transactions['unix_time'] = pd.to_datetime(
                        transactions['trans_date_trans_time']
                    ).astype(int) // 10**9
                except:
                    transactions['unix_time'] = 0
                    
            if 'merch_lat' not in transactions.columns:
                transactions['merch_lat'] = 40.7128
                
            if 'merch_long' not in transactions.columns:
                transactions['merch_long'] = -74.0060
            
            return transactions
            
        except FileNotFoundError:
            logger.error(f"Transaction file not found: {self.transaction_file}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading transaction data: {e}")
            raise
            
    def load_customer_data(self) -> pd.DataFrame:
        """
        Load customer data from CSV file.
        
        Returns:
            DataFrame with customer data
        """
        try:
            logger.info(f"Loading customers from {self.customer_file}")
            
            # Load CSV file
            customers = pd.read_csv(self.customer_file)
            
            logger.info(f"Loaded {len(customers)} customers")
            logger.info(f"Customer columns: {list(customers.columns)}")
            
            # Handle alternative column names
            if 'credit_card_number' in customers.columns and 'cc_num' not in customers.columns:
                customers['cc_num'] = customers['credit_card_number']
                
            # Add demographic columns if they don't exist
            default_demographics = {
                'gender': 'U',
                'city': 'Unknown',
                'state': 'Unknown',
                'zip': '00000',
                'job': 'Unknown',
                'dob': '1970-01-01',
                'city_pop': 100000
            }
            
            for col, default_val in default_demographics.items():
                if col not in customers.columns:
                    customers[col] = default_val
                    
            return customers
            
        except FileNotFoundError:
            logger.error(f"Customer file not found: {self.customer_file}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading customer data: {e}")
            raise
            
    def load_fraud_labels(self) -> pd.DataFrame:
        """
        Load fraud labels from JSON file.
        FIXED: Handles NaN/null values properly
        
        Returns:
            DataFrame with trans_num and is_fraud columns
        """
        try:
            logger.info(f"Loading fraud labels from {self.fraud_file}")
            
            # Load JSON file
            with open(self.fraud_file, 'r') as f:
                fraud_dict = json.load(f)
            
            # The JSON is a dictionary: {transaction_id: fraud_label}
            # Convert to DataFrame with proper structure
            fraud_df = pd.DataFrame(
                list(fraud_dict.items()),
                columns=['trans_num', 'is_fraud']
            )
            
            # Handle NaN, null, None values
            logger.info(f"Initial fraud labels shape: {fraud_df.shape}")
            
            # Check for null/NaN values
            null_count = fraud_df['is_fraud'].isnull().sum()
            if null_count > 0:
                logger.warning(f"Found {null_count} null/NaN fraud labels, treating as legitimate (0)")
            
            # Replace NaN/null with 0 (legitimate), then convert to int
            fraud_df['is_fraud'] = fraud_df['is_fraud'].fillna(0.0)
            
            # Handle any infinite values
            if np.isinf(fraud_df['is_fraud']).any():
                inf_count = np.isinf(fraud_df['is_fraud']).sum()
                logger.warning(f"Found {inf_count} infinite values, treating as legitimate (0)")
                fraud_df.loc[np.isinf(fraud_df['is_fraud']), 'is_fraud'] = 0.0
            
            # Convert to float first (to handle any remaining issues), then to int
            fraud_df['is_fraud'] = fraud_df['is_fraud'].astype(float).round().astype(int)
            
            # Ensure binary values (0 or 1)
            fraud_df.loc[fraud_df['is_fraud'] > 1, 'is_fraud'] = 1
            fraud_df.loc[fraud_df['is_fraud'] < 0, 'is_fraud'] = 0
            
            # Log statistics
            logger.info(f"Loaded {len(fraud_df)} fraud labels")
            fraud_count = fraud_df['is_fraud'].sum()
            legit_count = len(fraud_df) - fraud_count
            fraud_rate = fraud_df['is_fraud'].mean()
            
            logger.info(f"Fraud count: {fraud_count}")
            logger.info(f"Legitimate count: {legit_count}")
            logger.info(f"Fraud rate: {fraud_rate:.2%}")
            
            return fraud_df
            
        except FileNotFoundError:
            logger.error(f"Fraud file not found: {self.fraud_file}")
            return pd.DataFrame(columns=['trans_num', 'is_fraud'])
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading fraud labels: {e}")
            raise
            
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all data sources.
        
        Returns:
            Dictionary containing all dataframes
        """
        return {
            'transactions': self.load_transaction_data(),
            'customers': self.load_customer_data(),
            'fraud_labels': self.load_fraud_labels()
        }
    
    def get_merged_data(self) -> pd.DataFrame:
        """
        Load and merge all data sources properly.
        
        Returns:
            Merged DataFrame with all features and fraud labels
        """
        # Load all data
        transactions = self.load_transaction_data()
        customers = self.load_customer_data()
        fraud_labels = self.load_fraud_labels()
        
        # Merge fraud labels with transactions using trans_num
        if 'trans_num' in transactions.columns and 'trans_num' in fraud_labels.columns:
            logger.info("Merging fraud labels with transactions on trans_num...")
            transactions = transactions.merge(
                fraud_labels[['trans_num', 'is_fraud']],
                on='trans_num',
                how='left'
            )
            
            # Fill any missing fraud labels with 0 (legitimate)
            if transactions['is_fraud'].isnull().any():
                null_count = transactions['is_fraud'].isnull().sum()
                logger.warning(f"Filling {null_count} missing fraud labels with 0")
                transactions['is_fraud'] = transactions['is_fraud'].fillna(0).astype(int)
        else:
            logger.warning("Could not merge fraud labels - adding synthetic labels")
            np.random.seed(42)
            transactions['is_fraud'] = np.random.choice(
                [0, 1], 
                size=len(transactions), 
                p=[0.95, 0.05]
            )
        
        # Merge with customers
        if 'cc_num' in transactions.columns and 'cc_num' in customers.columns:
            logger.info("Merging customer data...")
            merged = transactions.merge(
                customers,
                on='cc_num',
                how='left'
            )
        else:
            merged = transactions
            
        logger.info(f"Final merged data: {len(merged)} rows, {len(merged.columns)} columns")
        logger.info(f"Final fraud rate: {merged['is_fraud'].mean():.2%}")
        
        return merged
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Validate the quality of loaded data.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        try:
            # Load data
            transactions = self.load_transaction_data()
            customers = self.load_customer_data()
            fraud_labels = self.load_fraud_labels()
            
            # Check data sizes
            validation_results['transaction_count'] = len(transactions)
            validation_results['customer_count'] = len(customers)
            validation_results['fraud_label_count'] = len(fraud_labels)
            
            # Check fraud label distribution
            if 'is_fraud' in fraud_labels.columns:
                fraud_dist = fraud_labels['is_fraud'].value_counts(normalize=True).to_dict()
                validation_results['fraud_distribution'] = {
                    'legitimate': fraud_dist.get(0, 0),
                    'fraudulent': fraud_dist.get(1, 0)
                }
                validation_results['fraud_rate'] = fraud_labels['is_fraud'].mean()
            
            validation_results['status'] = 'success'
            
        except Exception as e:
            validation_results['status'] = 'error'
            validation_results['error'] = str(e)
            
        return validation_results


# Test function
if __name__ == "__main__":
    # Test the data handler
    handler = RawDataHandler()
    
    print("Testing Fixed RawDataHandler with NaN handling...")
    print("-" * 50)
    
    # Test fraud labels loading
    print("Loading fraud labels...")
    try:
        fraud_labels = handler.load_fraud_labels()
        print(f"✅ Loaded {len(fraud_labels)} fraud labels")
        print(f"Columns: {list(fraud_labels.columns)}")
        if len(fraud_labels) > 0:
            print(f"\nFirst few entries:")
            print(fraud_labels.head())
            print(f"\nFraud distribution:")
            print(fraud_labels['is_fraud'].value_counts())
            print(f"Fraud rate: {fraud_labels['is_fraud'].mean():.2%}")
            print(f"Data types: {fraud_labels.dtypes}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "-" * 50)
    
    # Test merged data
    print("Testing merged data...")
    try:
        merged = handler.get_merged_data()
        print(f"✅ Merged data shape: {merged.shape}")
        print(f"Fraud rate in merged data: {merged['is_fraud'].mean():.2%}")
        print(f"NaN in is_fraud: {merged['is_fraud'].isnull().sum()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 50)
    print("Test complete!")