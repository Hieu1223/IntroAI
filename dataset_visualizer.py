import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

df = pd.read_csv('dataset/ames.csv')

print("=== Dataset Basic Statistics ===")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\nColumn types:")
print(df.dtypes.value_counts())

# Print missing values
print("\n=== Missing Values per Column ===")
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / df.shape[0]) * 100
missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Missing %': missing_percent})
missing_df = missing_df[missing_df['Missing Count'] > 0]  # Only show columns with missing values
print(missing_df.sort_values(by='Missing Count', ascending=False))



# Categorical stats
cat_cols = df.select_dtypes(include=['object']).columns
print(f"\nNumber of categorical columns: {len(cat_cols)}")
for col in cat_cols:
    top_values = df[col].value_counts().head(5)
    print(f"Top values for {col}:")
    print(top_values.to_string())
    print("---")

# Numerical stats
num_cols = df.select_dtypes(include=['number']).columns

# Compute descriptive statistics
num_summary = df[num_cols].describe().T[['count', 'min', '25%', '50%', '75%', 'max']]

# Sort by descending max, then ascending min
num_summary_sorted = num_summary.sort_values(by=['max', 'min'], ascending=[False, True])

# Remove the first 2 rows
num_summary_sorted = num_summary_sorted.iloc[2:]

# Display number of numerical columns and sorted summary
print(f"\nNumber of numerical columns: {len(num_cols)}\n")
print(num_summary_sorted)

cols_per_row = 7 
rows = math.ceil(len(cat_cols) / cols_per_row)

fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row*5, rows*4))
axes = axes.flatten() 

for i, col in enumerate(cat_cols):
    top_n = 10
    order = df[col].value_counts().index[:top_n]
    
    sns.countplot(
        data=df,
        x=col,
        order=order,
        ax=axes[i]
    )
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45)

# Hide any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()


output_file = "images/dataset/categorical_summary.png"
plt.savefig(output_file, dpi=300)
print(f"Saved categorical summary plot as {output_file}")
plt.close()  # Close the figure to free memory
