"""
Demo smooth and velocity
"""

import pandas as pd
import matplotlib.pyplot as plt
from stock01 import symble

# Define the file path
file_path = f'data/{symble}.csv'

# Read the data from the CSV file
df = pd.read_csv(file_path)

# Convert the 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Calculate a 9-point moving average of the close prices
df['Close_Smooth'] = df['Close'].rolling(window=9, center=True).mean()

# Calculate the velocity (difference) of the close prices
df['Velocity'] = df['Close_Smooth'].diff()

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot the close prices on the primary y-axis
ax1.plot(df['Date'], df['Close'], label='Close Price', color='blue')
ax1.plot(df['Date'], df['Close_Smooth'], label='9-Point Smooth', color='pink', linestyle='--')
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis to plot the velocity
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['Velocity'], label='Velocity', color='orange')
ax2.set_ylabel('Velocity', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Add a title and show the plot
plt.title('Close Price, 9-Point Smooth, and Velocity Over Time')
fig.tight_layout()

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Display the grid
plt.grid(True)

# Show the plot
plt.show()
