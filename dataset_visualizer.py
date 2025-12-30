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
print(f"\nNumber of numerical columns: {len(num_cols)}")
print(df[num_cols].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])

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
