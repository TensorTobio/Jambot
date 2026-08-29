import pandas as pd
from pathlib import Path

# Get the path relative to this script's directory
file_path = Path(__file__).parent / "public_set.jsonl"

# Open the file explicitly and pass the file handle
with open(file_path, "r", encoding="utf-8") as f:
    df = pd.read_json(f, lines=True)

# Print headers
print("--- Column Headers ---")
print(list(df.columns))

# Print first 4 rows
print("\n--- First 4 Lines ---")
print(df.head(1))

# unique_categories = df['categories'].explode().dropna().unique().tolist()
# print(unique_categories)