import pandas as pd
df = pd.read_csv("gold_data.csv")  # adjust filename to whatever you downloaded
print(df.shape)
print(df.columns.tolist())
df.info()
df.head()
df['Date'].head()
print(df['Date'].head().tolist())
