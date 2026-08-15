import pandas as pd

# Load the raw file
df = pd.read_csv("gold_data.csv")

# Convert Date column to real datetime — dayfirst=True because format is DD/MM/YYYY
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Sort chronologically (oldest first) — critical for time series
df = df.sort_values('Date')

# Set Date as index, since statsmodels expects this structure
df.set_index('Date', inplace=True)

# We only need Price for this project — drop the rest for now
df = df[['Price']]

print(df.shape)
df.head()
print(df.index.min(), df.index.max())
