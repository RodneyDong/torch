import pandas as pd
import matplotlib.pyplot as plt
smoothPoint = 10
# Load the data
file_path = "data/^GSPC.csv"
data = pd.read_csv(file_path)
# Ensure the data has a column for the date and close prices
# Adjust the column names if necessary
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
# Compute the moving average
# data['Smoothed'] = data['Close'].rolling(window=smoothPoint).mean()
data['Smoothed'] = data['Close'].rolling(window=smoothPoint, center=True).mean()
# data['Smoothed'] = data['Close'].rolling(window=smoothPoint).mean()
# Plot the original and smoothed data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Original Data')
plt.plot(data.index, data['Smoothed'], label=f'{smoothPoint}-Point Smoothed', color='orange')
# Add labels and legend
plt.title('Stock Data with 5-Point Smoothing')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
# Show the plot
plt.show()