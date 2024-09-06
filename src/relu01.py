import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

# Create an array of values ranging from -10 to 10
x = np.linspace(-10, 10, 100)

# Apply the ReLU function to the array
y = relu(x)

# Plot the result
plt.plot(x, y, label="ReLU(x)")
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("ReLU Activation Function")
plt.legend()
plt.grid()
plt.show()
