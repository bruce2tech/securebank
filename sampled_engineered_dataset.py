# Create a smaller sample locally
import pandas as pd
df = pd.read_csv('storage/datasets/dataset_engineered_raw.csv')
sample = df.sample(n=10000, random_state=42)
sample.to_csv('storage/datasets/dataset_sample.csv', index=False)
# This will be ~10MB instead of 1.18GB