#!/usr/bin/env python3
"""
Diagnostic script to understand the fraud labels structure
and fix the merging issue.
"""

import pandas as pd
import numpy as np
import json
import sys
sys.path.append('.')

from modules.data.raw_data_handler import RawDataHandler

print("=" * 60)
print("FRAUD LABELS DIAGNOSTIC")
print("=" * 60)
print()

# Load data
handler = RawDataHandler()

print("1. Loading fraud labels...")
print("-" * 40)
try:
    fraud_labels = handler.load_fraud_labels()
    print(f"✓ Loaded {len(fraud_labels)} fraud labels")
    print(f"  Columns: {list(fraud_labels.columns)}")
    print(f"  Shape: {fraud_labels.shape}")
    print()
    
    # Show first few rows
    print("First 3 rows:")
    print(fraud_labels.head(3))
    print()
    
    # Check is_fraud column
    if 'is_fraud' in fraud_labels.columns:
        print("Fraud distribution:")
        print(fraud_labels['is_fraud'].value_counts())
        print(f"Fraud rate: {fraud_labels['is_fraud'].mean():.2%}")
        print(f"NaN count: {fraud_labels['is_fraud'].isnull().sum()}")
    else:
        print("❌ No 'is_fraud' column found!")
        print("Available columns:", fraud_labels.columns.tolist())
except Exception as e:
    print(f"❌ Error loading fraud labels: {e}")

print()
print("2. Loading transactions...")
print("-" * 40)
try:
    transactions = handler.load_transaction_data()
    print(f"✓ Loaded {len(transactions)} transactions")
    print(f"  Columns: {list(transactions.columns)}")
    print(f"  Shape: {transactions.shape}")
    print()
    
    # Show first few rows
    print("First 3 rows:")
    print(transactions.head(3))
except Exception as e:
    print(f"❌ Error loading transactions: {e}")

print()
print("3. Checking for common merge keys...")
print("-" * 40)

if 'fraud_labels' in locals() and 'transactions' in locals():
    # Find common columns
    fraud_cols = set(fraud_labels.columns)
    trans_cols = set(transactions.columns)
    common_cols = fraud_cols & trans_cols
    
    print(f"Common columns: {common_cols}")
    
    # Check potential merge keys
    potential_keys = ['trans_num', 'transaction_id', 'cc_num', 'index', 'id']
    for key in potential_keys:
        in_fraud = key in fraud_labels.columns
        in_trans = key in transactions.columns
        if in_fraud or in_trans:
            print(f"  '{key}': fraud={in_fraud}, transactions={in_trans}")
    
    print()
    
    # Check if lengths match (for index-based merge)
    if len(fraud_labels) == len(transactions):
        print(f"✓ Same number of records ({len(fraud_labels)}), index-based merge possible")
    else:
        print(f"⚠ Different lengths: fraud={len(fraud_labels)}, transactions={len(transactions)}")
        print("  Index-based merge will require alignment")

print()
print("4. Testing merge strategies...")
print("-" * 40)

if 'fraud_labels' in locals() and 'transactions' in locals():
    # Test different merge strategies
    
    # Strategy 1: Direct index alignment
    print("Strategy 1: Index alignment")
    try:
        min_len = min(len(transactions), len(fraud_labels))
        test_trans = transactions.iloc[:100].copy()  # Test with first 100
        test_fraud = fraud_labels.iloc[:100].copy()
        test_trans['is_fraud'] = test_fraud['is_fraud'].values[:len(test_trans)]
        
        fraud_rate = test_trans['is_fraud'].mean()
        nan_count = test_trans['is_fraud'].isnull().sum()
        print(f"  ✓ Success: fraud_rate={fraud_rate:.2%}, NaN={nan_count}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Strategy 2: Merge by common column
    print("Strategy 2: Column-based merge")
    if common_cols:
        merge_col = list(common_cols)[0]
        print(f"  Trying to merge on '{merge_col}'")
        try:
            test_merged = transactions.iloc[:100].merge(
                fraud_labels[['is_fraud'] + [merge_col] if merge_col != 'is_fraud' else ['is_fraud']], 
                on=merge_col if merge_col != 'is_fraud' else None,
                how='left'
            )
            fraud_rate = test_merged['is_fraud'].mean()
            nan_count = test_merged['is_fraud'].isnull().sum()
            print(f"  ✓ Success: fraud_rate={fraud_rate:.2%}, NaN={nan_count}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    else:
        print("  ⚠ No common columns for merge")

print()
print("5. Recommended solution...")
print("-" * 40)

# Determine best approach
if 'fraud_labels' in locals() and 'transactions' in locals():
    if len(fraud_labels) == len(transactions):
        print("✅ Use index-based alignment:")
        print("   transactions['is_fraud'] = fraud_labels['is_fraud'].values")
    elif common_cols:
        merge_col = list(common_cols)[0]
        print(f"✅ Use column-based merge on '{merge_col}':")
        print(f"   transactions.merge(fraud_labels, on='{merge_col}', how='left')")
    else:
        print("⚠ Use truncated index alignment:")
        print("   min_len = min(len(transactions), len(fraud_labels))")
        print("   transactions.iloc[:min_len]['is_fraud'] = fraud_labels['is_fraud'].values[:min_len]")

print()
print("6. Checking actual fraud file structure...")
print("-" * 40)

# Read the raw JSON file to understand its structure
try:
    with open('data_sources/fraud_release.json', 'r') as f:
        fraud_json = json.load(f)
    
    if isinstance(fraud_json, dict):
        print(f"JSON is a dictionary with keys: {list(fraud_json.keys())[:10]}")
        # Show structure of first item
        for key in list(fraud_json.keys())[:1]:
            print(f"  '{key}': {fraud_json[key]}")
    elif isinstance(fraud_json, list):
        print(f"JSON is a list with {len(fraud_json)} items")
        if fraud_json:
            print(f"  First item: {fraud_json[0]}")
    else:
        print(f"JSON is of type: {type(fraud_json)}")
        
except Exception as e:
    print(f"Error reading fraud JSON: {e}")

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)