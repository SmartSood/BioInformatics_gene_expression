import pandas as pd
from pathlib import Path

# Input and output file paths
input_file = Path("/Users/smarthsood/Downloads/GSE28375 2 copy.csv")
output_file = Path("/Users/smarthsood/Downloads/output_transposed.csv")

# Read CSV
df = pd.read_csv(input_file, index_col=0)

# Transpose rows and columns
df_transposed = df.T

# Save WITHOUT row names/index
df_transposed.to_csv(output_file, index=False)

print("Original shape:", df.shape)
print("Transposed shape:", df_transposed.shape)
print(f"Transposed CSV saved to: {output_file}")