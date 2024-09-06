import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('data/linear_data_with_deviation.csv')
x = data['x'].values
y = data['y'].values

# # Normalize the data
# x_mean = x.mean()
# x_std = x.std()
# y_mean = y.mean()
# y_std = y.std()

# x_normalized = (x - x_mean) / x_std
# y_normalized = (y - y_mean) / y_std

# Prepare the data
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Get the model's parameters
for name, param in model.named_parameters():
    if name == 'linear.weight':
        slope_normalized = param.item()
    if name == 'linear.bias':
        intercept_normalized = param.item()

# Convert slope and intercept back to original scale
slope = slope_normalized * (y / x)
intercept = intercept_normalized * y_std + y_mean - slope * x_mean

print(f'Slope: {slope}')
print(f'Intercept: {intercept}')

# Plot the original data
plt.scatter(x, y, color='blue', label='Original Data')

# Plot the fitted line
y_pred_normalized = model(x_tensor).detach().numpy()
y_pred = y_pred_normalized * y_std + y_mean
plt.plot(x, y_pred, color='red', label='Fitted Line')

plt.xlabel('Column1')
plt.ylabel('Column2')
plt.title('Linear Fit using Neural Network')
plt.legend()
plt.show()
