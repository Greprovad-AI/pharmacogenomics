import sys
import torch
import transformers
import pandas as pd
import os

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("Pandas version:", pd.__version__)
print("Current working directory:", os.getcwd())

# Try to read the CSV file
csv_path = os.path.join(os.getcwd(), "data", "genome_sequences", "hg38", "sequences_hg38_200b_verysmall.csv")
print("\nTesting file access:")
print(f"CSV path: {csv_path}")
print(f"File exists: {os.path.exists(csv_path)}")

if os.path.exists(csv_path):
    print("\nReading first few lines of CSV:")
    df = pd.read_csv(csv_path)
    print(f"Number of sequences: {len(df)}")
    print("\nFirst sequence:")
    print(df["sequence"].iloc[0])
