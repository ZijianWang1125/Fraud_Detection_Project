import pandas as pd

# Load the original credit card dataset
df = pd.read_csv("../data/creditcard_2023.csv")

# Sample 100 rows for validation, keep Class column
validation_sample = df.sample(n=100, random_state=42)

# Save to new file
validation_sample.to_csv("../data/validation_sample.csv", index=False)

print("Generated 'validation_sample.csv' with 100 rows and true Class labels.")
