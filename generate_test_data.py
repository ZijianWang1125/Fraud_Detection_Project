import pandas as pd
import numpy as np

np.random.seed(42)
num_samples = 100

# Create random V1-V28 features and Amount
data = {f'V{i}': np.random.normal(0, 1, num_samples) for i in range(1, 29)}
data['Amount'] = np.random.uniform(1, 5000, num_samples)

df = pd.DataFrame(data)
df.to_csv("test_transactions.csv", index=False)
