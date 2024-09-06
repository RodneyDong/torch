"""
Use test data, pickup any row by index, and predict the classified label.
"""
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from stock import NeuralNetwork, file_path, getDataSet, labels, window

# Load the trained model
model = NeuralNetwork()
model.load_state_dict(torch.load("outputs/stockTestModel1.pth"))
model.eval()

# Load the dataset
training_data, test_data = getDataSet(file_path)

# Define the index for the test data
index = 6  # Change this index to the desired test sample
# Get the specific test sample

stock, decision = test_data[index][0], test_data[index][1]

# Get the actual label from the decision tensor
actual_label = int(decision)

# Reshape the stock data to match the input shape expected by the model
stock = stock.reshape(1, 6, window)

# Predict using the model
with torch.no_grad():
    pred = model(stock)
    predicted_label = pred[0].argmax(0).item()

# Output the prediction and actual result
print(f'Index: {index}')
print(f'Predicted: "{labels[predicted_label]}", Actual: "{labels[actual_label]}"')