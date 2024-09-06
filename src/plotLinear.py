import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
csv_file_path = 'data/linear_data_with_deviation.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Replace 'Column1' and 'Column2' with the actual column names in your CSV
x = df['x']
y = df['y']

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, marker='o')

# Add title and labels
plt.title('Plot of X vs Y')
plt.xlabel('X')
plt.ylabel('Y')

# Show the plot
plt.show()