import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
from PIL import Image

# Define the neural network (ensure this matches your model architecture)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Load the pre-trained model
model = NeuralNetwork()
model.load_state_dict(torch.load("outputs/fashion_model.pth"))
model.eval()

# Load the Fashion-MNIST test dataset
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

# Select a random test sample
index = random.randint(0, len(test_data) - 1)
x, y = test_data[index]
print(f"Index: {index}, Label: {y}")

# Display the selected test image
plt.figure()
plt.imshow(x.squeeze(), cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()

# Use the model to predict the label
with torch.no_grad():
    pred = model(x.unsqueeze(0))  # Unsqueeze to add batch dimension
    predicted = pred.argmax(1).item()  # Get the index of the highest logit
    actual = y  # Actual label

print(f'Predicted: "{predicted}", Actual: "{actual}"')
