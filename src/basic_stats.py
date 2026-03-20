import pandas as pd

# Load dataset (local file)
data = pd.read_csv("train.csv")

# Show first rows
print("First 5 rows:")
print(data.head())

# Dataset info
print("\nDataset Info:")
print(data.info())

# Statistical summary (numeric columns)
print("\nStatistical Summary:")
print(data.describe())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Value counts (example: survival)
print("\nSurvival Count:")
print(data["Survived"].value_counts())

print("\n Column Names:")
print(data.columns.tolist())

print("\n Dataset Shape:")
print(data.shape)