import pandas as pd

df = pd.read_csv('dataset/kc_final.csv')

# Check for missing values
missing = df.isnull().sum()

# Filter columns that have at least one NaN
missing_cols = missing[missing > 0]

print("Columns with missing values:")
print(missing_cols)