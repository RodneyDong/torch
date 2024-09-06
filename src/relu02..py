import numpy as np
import matplotlib.pyplot as plt

# Generate a sine wave
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = np.sin(x)

# Apply ReLU function
relu_y = np.maximum(0, y)

# Plot the original sine wave and the ReLU-transformed sine wave
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(x, y, label='Sine Wave')
plt.title('Original Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, relu_y, label='ReLU Applied to Sine Wave', color='orange')
plt.title('Sine Wave with ReLU Activation')
plt.xlabel('x')
plt.ylabel('ReLU(sin(x))')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
