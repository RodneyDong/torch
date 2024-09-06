"""
load stock data from internet and save them into a csv file
pip install yfinance
"""
import datetime
import yfinance as yf
import os

symble = "MSFT"
symble = "^GSPC"
# Download data
if __name__ == "__main__":
    # Define backtest range
    START = datetime.datetime(2020, 1, 1)
    END = datetime.datetime(2024, 1, 1)
    df = yf.download(symble, START, END)
    print(df)
    # Output DataFrame properties
    print("\nDataFrame Properties:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"Index: {df.index}")
    print(f"Data Types:\n{df.dtypes}")
    # Select the 'Close' column
    selected_columns = df[['Close']]
    # Reset the index to include the Date as a column
    selected_columns_reset = selected_columns.reset_index()
    # # Convert the Date column to Unix timestamp
    # selected_columns_reset['Date'] = selected_columns_reset['Date'].apply(lambda x: int(x.timestamp()))
    # Ensure the 'data' directory exists
    os.makedirs('data', exist_ok=True)
    # Save the selected columns to a CSV file
    file_path = f'data/{symble}.csv'
    selected_columns_reset.to_csv(file_path, index=False)
    print(f"Selected columns with Unix timestamp saved to {file_path}")