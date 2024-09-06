import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np

# Generate some dummy data
np.random.seed(0)
X = np.random.rand(100, 1)  # 100 samples, 1 feature
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.1  # Linear relationship with noise

# Define a simple neural network model
model = Sequential([
    Dense(1, input_dim=1)
])

# Compile the model with different learning rates
learning_rates = [0.01, 0.1, 0.001]

# Train the model with different learning rates and plot the loss
for lr in learning_rates:
    print(f'\nTraining with learning rate: {lr}')
    model.compile(optimizer=SGD(learning_rate=lr), loss='mse')
    history = model.fit(X, y, epochs=100, verbose=0)
    
    # Plot the loss
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'], label=f'LR={lr}')
    
X = np.random.rand(100, 1)  # 100 samples, 1 feature
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.1  # Linear relationship with noise
plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epochs for different learning rates')
plt.show()
