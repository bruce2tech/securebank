#!/usr/bin/env python3
"""
Create sample test data for SecureBank system testing.
This script generates minimal but realistic test datasets if real data is not available.
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

def create_sample_customers(n_customers=100):
    """Create sample customer data."""
    np.random.seed(42)
    
    first_names = ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 'Diana', 'Eve', 'Frank']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia']
    states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA']
    jobs = ['Engineer', 'Teacher', 'Doctor', 'Artist', 'Manager', 'Analyst', 'Designer']
    
    customers = []
    for i in range(n_customers):
        cc_num = 4000000000000000 + i  # Simple sequential credit card numbers
        customers.append({
            'cc_num': cc_num,
            'index': i,
            'first': np.random.choice(first_names),
            'last': np.random.choice(last_names),
            'sex': np.random.choice(['M', 'F']),
            'street': f"{np.random.randint(1, 9999)} Main St",
            'city': np.random.choice(cities),
            'state': np.random.choice(states),
            'zip': np.random.randint(10000, 99999),
            'lat': np.random.uniform(25, 50),  # US latitude range
            'long': np.random.uniform(-125, -65),  # US longitude range
            'city_pop': np.random.randint(10000, 1000000),
            'job': np.random.choice(jobs),
            'dob': (datetime.now() - timedelta(days=np.random.randint(25*365, 75*365))).strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(customers)

def create_sample_transactions(customers, n_transactions=1000):
    """Create sample transaction data."""
    np.random.seed(42)
    
    merchants = ['Walmart', 'Target', 'Amazon', 'Starbucks', 'McDonalds', 'Shell', 'Home Depot', 'Best Buy']
    categories = ['grocery_pos', 'gas_transport', 'shopping_net', 'food_dining', 'entertainment', 'health_fitness']
    
    transactions = []
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(n_transactions):
        # Random customer
        customer = customers.sample(1).iloc[0]
        
        # Random transaction time
        trans_date = start_date + timedelta(
            days=np.random.randint(0, 365),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        )
        
        transactions.append({
            'trans_date_trans_time': trans_date.strftime('%Y-%m-%d %H:%M:%S'),
            'cc_num': customer['cc_num'],
            'merchant': np.random.choice(merchants),
            'category': np.random.choice(categories),
            'amt': round(np.random.lognormal(3, 1), 2),  # Log-normal distribution for amounts
            'first': customer['first'],
            'last': customer['last'],
            'gender': customer['sex'],
            'street': customer['street'],
            'city': customer['city'],
            'state': customer['state'],
            'zip': customer['zip'],
            'lat': customer['lat'],
            'long': customer['long'],
            'city_pop': customer['city_pop'],
            'job': customer['job'],
            'dob': customer['dob'],
            'trans_num': f"T{i:06d}",
            'unix_time': int(trans_date.timestamp()),
            'merch_lat': customer['lat'] + np.random.normal(0, 0.1),
            'merch_long': customer['long'] + np.random.normal(0, 0.1)
        })
    
    return pd.DataFrame(transactions)

def create_sample_fraud_labels(transactions, fraud_rate=0.05):
    """Create sample fraud labels."""
    np.random.seed(42)
    
    n_fraudulent = int(len(transactions) * fraud_rate)
    fraud_indices = np.random.choice(len(transactions), n_fraudulent, replace=False)
    
    fraud_dict = {}
    for i, trans_num in enumerate(transactions['trans_num']):
        fraud_dict[trans_num] = 1 if i in fraud_indices else 0
    
    return fraud_dict

def main():
    """Generate all test data files."""
    print("🔧 Creating sample test data...")
    
    # Create data_sources directory
    os.makedirs('data_sources', exist_ok=True)
    
    # Generate data
    print("📊 Generating customers...")
    customers = create_sample_customers(n_customers=100)
    
    print("💳 Generating transactions...")
    transactions = create_sample_transactions(customers, n_transactions=1000)
    
    print("🚨 Generating fraud labels...")
    fraud_labels = create_sample_fraud_labels(transactions, fraud_rate=0.05)
    
    # Save data files
    print("💾 Saving data files...")
    
    # Customer data (CSV)
    customers.to_csv('data_sources/customer_release.csv', index=False)
    print(f"   ✓ customer_release.csv ({len(customers)} customers)")
    
    # Transaction data (Parquet)
    transactions.to_parquet('data_sources/transactions_release.parquet', engine='pyarrow')
    print(f"   ✓ transactions_release.parquet ({len(transactions)} transactions)")
    
    # Fraud labels (JSON)
    with open('data_sources/fraud_release.json', 'w') as f:
        json.dump(fraud_labels, f)
    print(f"   ✓ fraud_release.json ({sum(fraud_labels.values())} fraudulent)")
    
    print("✅ Sample test data created successfully!")
    print(f"   - Customers: {len(customers)}")
    print(f"   - Transactions: {len(transactions)}")
    print(f"   - Fraud rate: {sum(fraud_labels.values())/len(fraud_labels)*100:.1f}%")
    print()
    print("   Now run the SecureBank tests!")

if __name__ == "__main__":
    main()