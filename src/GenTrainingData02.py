import pandas as pd
from GenTrainingData import td_file
# Read the CSV file into a DataFrame
data = pd.read_csv(td_file, header=None)

n = 3
#get nth row in the data

first_row = data.iloc[n].tolist()
print(first_row)