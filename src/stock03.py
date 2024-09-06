"""
draw boolinger and MACD lines
refer to ../doc/boolinger.md
"""

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Define the ticker and date range
ticker = '^GSPC'
start_date = '2023-01-01'
end_date = '2023-12-31'

# Fetch data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Display the first few rows of the data
print(data.head())

# Calculate the short-term and long-term moving averages
short_ema = data['Close'].ewm(span=12, adjust=False).mean()  # 12-day EMA Exponential Mean Average
long_ema = data['Close'].ewm(span=26, adjust=False).mean()  # 26-day EMA

# Calculate the MACD line
macd = short_ema - long_ema

# Calculate the Signal line
signal = macd.ewm(span=9, adjust=False).mean()

# Calculate the MACD histogram
histogram = macd - signal

# Add MACD, Signal line, and histogram to the DataFrame
data['MACD'] = macd
data['Signal'] = signal
data['Histogram'] = histogram

# Calculate Bollinger Bands
window = 20
data['Rolling Mean'] = data['Close'].rolling(window=window).mean()
data['Bollinger High'] = data['Rolling Mean'] + 2 * data['Close'].rolling(window=window).std()
data['Bollinger Low'] = data['Rolling Mean'] - 2 * data['Close'].rolling(window=window).std()

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 10))

# Plot the closing prices with Bollinger Bands
ax1.plot(data['Close'], label='Close Price', color='blue')
ax1.plot(data['Rolling Mean'], label='Rolling Mean (20 days)', color='orange', linestyle='--')
ax1.plot(data['Bollinger High'], label='Bollinger High', linestyle='--', color='red')
ax1.plot(data['Bollinger Low'], label='Bollinger Low', linestyle='--', color='green')
ax1.set_title('S&P 500 Closing Prices with Bollinger Bands')
ax1.set_ylabel('Price')
ax1.legend()

# Plot the MACD and Signal line
ax2.plot(data['MACD'], label='MACD', color='blue')
ax2.plot(data['Signal'], label='Signal Line', color='red')

# Plot the MACD histogram with colors
pos_histogram = data['Histogram'].where(data['Histogram'] > 0)
neg_histogram = data['Histogram'].where(data['Histogram'] <= 0)
ax2.bar(data.index, pos_histogram, label='Positive Histogram', color='green', alpha=0.6)
ax2.bar(data.index, neg_histogram, label='Negative Histogram', color='red', alpha=0.6)

ax2.set_title('MACD')
ax2.set_xlabel('Date')
ax2.set_ylabel('Value')
ax2.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()
