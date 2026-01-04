"""
Synthetic Data Generator for SecureBank Fraud Detection
Clean, standalone module for generating realistic synthetic transaction data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

class SyntheticDataGenerator:
    """
    Generates realistic synthetic fraud detection datasets
    """
    
    def __init__(self):
        """Initialize the synthetic data generator with realistic parameters"""
        
        # Realistic merchant names
        self.merchants = [
            'Amazon', 'Walmart', 'Target', 'Starbucks', 'McDonalds', 'Shell', 'Exxon',
            'Home Depot', 'Best Buy', 'CVS', 'Walgreens', 'Kroger', 'Safeway',
            'Costco', 'IKEA', 'Macy\'s', 'Nike', 'Apple Store', 'Google Play',
            'Uber', 'Lyft', 'Netflix', 'Spotify', 'Steam', 'PayPal', 'Square'
        ]
        
        # Transaction categories from real fraud detection datasets
        self.categories = [
            'grocery_pos',      # Grocery point-of-sale
            'gas_transport',    # Gas stations and transportation
            'entertainment',    # Movies, games, etc.
            'food_dining',      # Restaurants and dining
            'health_fitness',   # Health and fitness
            'home',            # Home improvement
            'kids_pets',       # Kids and pets
            'misc_pos',        # Miscellaneous point-of-sale
            'personal_care',   # Personal care items
            'shopping_pos',    # Shopping point-of-sale
            'shopping_net',    # Online shopping
            'travel'           # Travel and hotels
        ]
        
        # US geographic boundaries (approximate)
        self.lat_range = (25.0, 49.0)    # Southern to Northern US
        self.long_range = (-125.0, -66.0)  # Western to Eastern US
        
        # Major US cities for realistic merchant locations
        self.major_cities = [
            (40.7128, -74.0060),   # New York
            (34.0522, -118.2437),  # Los Angeles
            (41.8781, -87.6298),   # Chicago
            (29.7604, -95.3698),   # Houston
            (33.4484, -112.0740),  # Phoenix
            (39.9526, -75.1652),   # Philadelphia
            (29.4241, -98.4936),   # San Antonio
            (32.7767, -96.7970),   # Dallas
            (37.3382, -121.8863),  # San Jose
            (30.2672, -97.7431)    # Austin
        ]
    
    def generate_synthetic_dataset(self, size=10000, fraud_rate=0.05, random_seed=None, save_path=None):
        """
        Generate a realistic synthetic fraud detection dataset
        
        Args:
            size (int): Number of transactions to generate
            fraud_rate (float): Proportion of fraudulent transactions (0.0-1.0)
            random_seed (int): Random seed for reproducibility
            save_path (str): Optional path to save the dataset
            
        Returns:
            pd.DataFrame: Generated synthetic dataset
        """
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            
        print(f"🔧 Generating {size:,} synthetic transactions (fraud rate: {fraud_rate:.1%})")
        
        # Generate customer base (credit card numbers)
        num_customers = min(size // 10, 2000)  # Reasonable number of unique customers
        cc_numbers = self._generate_credit_card_numbers(num_customers)
        
        # Generate transaction data
        transactions = []
        base_time = datetime(2023, 1, 1)
        
        for i in range(size):
            transaction = self._generate_single_transaction(
                i, cc_numbers, base_time, fraud_rate
            )
            transactions.append(transaction)
            
            # Progress indicator for large datasets
            if size > 20000 and i % 10000 == 0:
                print(f"   📊 Generated {i:,} transactions...")
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        
        # Validate and clean data
        df = self._validate_and_clean_dataset(df)
        
        # Report generation statistics
        actual_fraud_rate = df['is_fraud'].mean()
        print(f"✅ Generated dataset: {len(df):,} records")
        print(f"   📈 Actual fraud rate: {actual_fraud_rate:.1%}")
        print(f"   👥 Unique customers: {df['cc_num'].nunique():,}")
        print(f"   🏪 Unique merchants: {df['merchant'].nunique()}")
        print(f"   📅 Date range: {df['trans_date_trans_time'].min()} to {df['trans_date_trans_time'].max()}")
        
        # Save if path provided
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"   💾 Dataset saved to: {save_path}")
            
        return df
    
    def _generate_credit_card_numbers(self, count):
        """Generate realistic credit card numbers"""
        # Use different card prefixes for realism
        prefixes = [4000, 4111, 4532, 5555, 5105]  # Visa/Mastercard test ranges
        
        cc_numbers = []
        for _ in range(count):
            prefix = random.choice(prefixes)
            # Generate remaining digits
            remaining = random.randint(100000000000, 999999999999)
            cc_number = int(f"{prefix}{remaining}")
            cc_numbers.append(cc_number)
            
        return cc_numbers
    
    def _generate_single_transaction(self, transaction_id, cc_numbers, base_time, fraud_rate):
        """Generate a single realistic transaction"""
        
        # Determine if this transaction is fraudulent
        is_fraud = random.random() < fraud_rate
        
        # Generate timestamp with realistic patterns
        transaction_time = self._generate_transaction_time(base_time, is_fraud)
        
        # Select customer (some have multiple transactions)
        cc_num = random.choice(cc_numbers)
        
        # Generate merchant and category
        merchant = random.choice(self.merchants)
        category = random.choice(self.categories)
        
        # Generate location
        merch_lat, merch_long = self._generate_merchant_location(is_fraud)
        
        # Generate transaction amount
        amount = self._generate_transaction_amount(category, is_fraud)
        
        # Apply fraud patterns
        if is_fraud:
            merchant, category, merch_lat, merch_long, amount = self._apply_fraud_patterns(
                merchant, category, merch_lat, merch_long, amount, transaction_time
            )
        
        return {
            'trans_date_trans_time': transaction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'cc_num': cc_num,
            'unix_time': int(transaction_time.timestamp()),
            'merchant': merchant,
            'category': category,
            'amt': round(amount, 2),
            'merch_lat': round(merch_lat, 4),
            'merch_long': round(merch_long, 4),
            'is_fraud': int(is_fraud)
        }
    
    def _generate_transaction_time(self, base_time, is_fraud):
        """Generate realistic transaction timestamp"""
        
        # Random day within a year
        days_offset = random.randint(0, 365)
        
        if is_fraud:
            # Fraudulent transactions more likely at unusual hours
            if random.random() < 0.4:  # 40% of fraud occurs at night
                hour = random.choice([23, 0, 1, 2, 3, 4])
            else:
                hour = random.randint(0, 23)
        else:
            # Legitimate transactions follow normal business patterns
            hour_weights = [1, 1, 1, 1, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2]
            hour = random.choices(range(24), weights=hour_weights)[0]
        
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return base_time + timedelta(
            days=days_offset,
            hours=hour,
            minutes=minute,
            seconds=second
        )
    
    def _generate_merchant_location(self, is_fraud):
        """Generate merchant location coordinates"""
        
        if is_fraud and random.random() < 0.3:
            # Some fraudulent transactions from unusual locations
            lat = random.uniform(20.0, 60.0)  # Broader range
            long = random.uniform(-140.0, -50.0)
        else:
            # Most transactions near major cities
            if random.random() < 0.6:  # 60% near major cities
                city_lat, city_long = random.choice(self.major_cities)
                # Add some noise around the city
                lat = city_lat + random.gauss(0, 0.5)
                long = city_long + random.gauss(0, 0.5)
            else:
                # Random US location
                lat = random.uniform(*self.lat_range)
                long = random.uniform(*self.long_range)
        
        return lat, long
    
    def _generate_transaction_amount(self, category, is_fraud):
        """Generate realistic transaction amount based on category and fraud status"""
        
        # Base amounts by category (log-normal parameters)
        category_params = {
            'grocery_pos': (3.2, 0.6),      # ~$24 average
            'gas_transport': (3.4, 0.5),    # ~$30 average
            'entertainment': (2.8, 0.8),    # ~$16 average
            'food_dining': (2.9, 0.7),      # ~$18 average
            'health_fitness': (3.5, 0.9),   # ~$30 average
            'home': (4.2, 1.0),             # ~$66 average
            'kids_pets': (2.7, 0.6),        # ~$15 average
            'misc_pos': (3.0, 1.2),         # ~$20 average
            'personal_care': (2.5, 0.5),    # ~$12 average
            'shopping_pos': (3.3, 0.8),     # ~$27 average
            'shopping_net': (3.6, 1.0),     # ~$36 average
            'travel': (4.8, 1.2)            # ~$121 average
        }
        
        mean, sigma = category_params.get(category, (3.2, 0.8))
        
        if is_fraud:
            # Fraudulent transactions tend to be higher
            mean += 0.8  # Increase mean for fraud
            if random.random() < 0.2:  # 20% of fraud is very high amounts
                amount = random.uniform(500, 2000)
            else:
                amount = np.random.lognormal(mean, sigma)
        else:
            amount = np.random.lognormal(mean, sigma)
        
        # Ensure reasonable bounds
        return max(0.01, min(amount, 5000.0))
    
    def _apply_fraud_patterns(self, merchant, category, lat, long, amount, transaction_time):
        """Apply realistic fraud patterns to transactions"""
        
        # Pattern 1: Card testing (small amounts)
        if random.random() < 0.15:  # 15% of fraud
            amount = random.uniform(0.01, 5.00)
            
        # Pattern 2: Unusual merchant for high amounts
        if amount > 200 and random.random() < 0.3:
            merchant = random.choice(['UNKNOWN_MERCHANT', 'ONLINE_PURCHASE', 'CASH_ADVANCE'])
            
        # Pattern 3: Multiple categories (fraudsters try different things)
        if random.random() < 0.2:
            category = random.choice(self.categories)
            
        # Pattern 4: Round amounts (automated fraud)
        if random.random() < 0.25:
            amount = round(amount / 10) * 10  # Round to nearest $10
            
        return merchant, category, lat, long, amount
    
    def _validate_and_clean_dataset(self, df):
        """Validate and clean the generated dataset"""
        
        # Ensure no negative amounts
        df['amt'] = df['amt'].clip(lower=0.01)
        
        # Ensure valid coordinates
        df['merch_lat'] = df['merch_lat'].clip(-90, 90)
        df['merch_long'] = df['merch_long'].clip(-180, 180)
        
        # Sort by timestamp
        df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
        
        return df
    
    def generate_and_save_dataset(self, size=10000, fraud_rate=0.05, random_seed=None, 
                                 filename=None, storage_dir="securebank/storage/datasets"):
        """
        Generate and save a synthetic dataset with automatic naming
        
        Args:
            size: Number of records
            fraud_rate: Fraud percentage
            random_seed: Random seed
            filename: Custom filename (optional)
            storage_dir: Directory to save dataset
            
        Returns:
            tuple: (DataFrame, file_path)
        """
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"dataset_synthetic_{timestamp}.csv"
            
        file_path = os.path.join(storage_dir, filename)
        
        # Generate dataset
        df = self.generate_synthetic_dataset(
            size=size,
            fraud_rate=fraud_rate,
            random_seed=random_seed,
            save_path=file_path
        )
        
        return df, file_path

# Convenience function for quick dataset generation
def generate_fraud_dataset(size=10000, fraud_rate=0.05, random_seed=42, save_path=None):
    """
    Quick function to generate a fraud detection dataset
    
    Args:
        size: Number of transactions
        fraud_rate: Proportion of fraudulent transactions
        random_seed: Random seed for reproducibility
        save_path: Path to save dataset
        
    Returns:
        pd.DataFrame: Generated dataset
    """
    generator = SyntheticDataGenerator()
    return generator.generate_synthetic_dataset(size, fraud_rate, random_seed, save_path)

# Example usage
if __name__ == "__main__":
    # Test the generator
    print("🧪 Testing Synthetic Data Generator")
    
    generator = SyntheticDataGenerator()
    
    # Generate small test dataset
    test_df = generator.generate_synthetic_dataset(size=1000, fraud_rate=0.1, random_seed=42)
    
    print(f"\n📊 Dataset Overview:")
    print(f"Shape: {test_df.shape}")
    print(f"Columns: {list(test_df.columns)}")
    print(f"Fraud rate: {test_df['is_fraud'].mean():.1%}")
    print(f"Amount range: ${test_df['amt'].min():.2f} - ${test_df['amt'].max():.2f}")
    print(f"Date range: {test_df['trans_date_trans_time'].min()} to {test_df['trans_date_trans_time'].max()}")
    
    print(f"\n📈 Sample transactions:")
    print(test_df.head())
    
    print(f"\n🔍 Fraud examples:")
    print(test_df[test_df['is_fraud'] == 1].head())