import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv('dataset/ames.csv')

# -----------------------------
# 2. Detect categorical columns
# -----------------------------
cat_cols = df.select_dtypes(include=['object']).columns
num_cols = len(cat_cols)
print(f"Number of categorical columns: {num_cols}")

# -----------------------------
# 3. Grid layout for plotting
# -----------------------------
cols_per_row = 4  # adjust to your screen or image size
rows = math.ceil(num_cols / cols_per_row)

fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row*5, rows*4))
axes = axes.flatten()  # flatten 2D array for easy indexing

# -----------------------------
# 4. Plot each categorical column
# -----------------------------
for i, col in enumerate(cat_cols):
    # Optional: limit to top 10 values to avoid overcrowding
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

# -----------------------------
# 5. Save figure as high-resolution image
# -----------------------------
output_file = "images/dataset/categorical_summary.png"
plt.savefig(output_file, dpi=300)
print(f"Saved categorical summary plot as {output_file}")
plt.close()  # Close the figure to free memory
