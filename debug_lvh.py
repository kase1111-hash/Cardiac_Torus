import pandas as pd
from pathlib import Path

df = pd.read_csv(r"G:\EchoNet-LVH\MeasurementsList.csv")
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nFirst 10 rows:")
print(df.head(10).to_string())
print("\nUnique Calc values:", df['Calc'].unique())
print("\nCalc value counts:")
print(df['Calc'].value_counts())
print("\nSample HashedFileName values:")
print(df['HashedFileName'].head(20).tolist())
print("\nUnique videos:", df['HashedFileName'].nunique())

# Check what actual video files look like
batch1 = Path(r"G:\EchoNet-LVH\Batch1")
files = list(batch1.iterdir())[:10]
print("\nSample files in Batch1:")
for f in files:
    print(f"  {f.name}")
