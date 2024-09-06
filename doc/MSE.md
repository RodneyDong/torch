Mean Square Error (MSE) is a common loss function used in regression tasks to measure the average of the squares of the errors—that is, the average squared difference between the actual and predicted values. It provides a way to quantify how close a model's predictions are to the actual values.

### Formula

For a set of predictions \(\hat{y}\) and actual values \(y\), the Mean Square Error is calculated as:

\[ \text{MSE}(\hat y_i, x_i, w_i, bi) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 \]

where:
- \( n \) is the number of data points,
- \( \hat{y}_i \) is the predicted value for the \(i\)-th data point,
- \( y_i \) is the actual value for the \(i\)-th data point.

In linear model, where $\hat y=w_i x_i +b_i$
### Key Characteristics

1. **Non-Negative**: MSE is always non-negative because it is a sum of squared differences. The lowest possible value is 0, which indicates perfect predictions.
2. **Sensitivity to Outliers**: Because it squares the differences, MSE is more sensitive to large errors. This can be useful if you want to heavily penalize larger errors.
3. **Differentiable**: MSE is differentiable, which makes it suitable for optimization algorithms that rely on gradient descent.

### Use in Machine Learning

In machine learning, MSE is commonly used as a loss function for training regression models. It helps guide the optimization algorithm (such as gradient descent) to minimize the error between the model's predictions and the actual data.

### Example in Python with PyTorch

Here's a simple example demonstrating how to use MSE as a loss function in PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# Create dummy data
X = torch.randn(100, 1)  # 100 samples, each with 1 feature
y = 3 * X + 2 + torch.randn(100, 1)  # y = 3x + 2 + noise

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(X)  # Forward pass
    loss = criterion(outputs, y)  # Compute the loss (MSE)
    loss.backward()  # Backward pass
    optimizer.step()  # Update the weights
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

print('Training complete!')
```

### Explanation:
1. **Linear Regression Model**: A simple linear model with one input and one output.
2. **Dummy Data**: Generated data based on the equation \( y = 3x + 2 \) with some added noise.
3. **Loss Function**: MSE is used as the loss function (`nn.MSELoss()`).
4. **Optimizer**: Stochastic Gradient Descent (SGD) is used to optimize the model parameters.
5. **Training Loop**: The model is trained for a specified number of epochs, with the loss printed every 10 epochs.

This example demonstrates how to set up a regression model in PyTorch and use MSE to measure and minimize the error between the predicted and actual values.