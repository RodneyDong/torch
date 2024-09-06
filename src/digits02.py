import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from digits01 import NeuralNetwork, test_data
import matplotlib.pyplot as plt
import numpy as np
import random 

model = NeuralNetwork()
model.load_state_dict(torch.load("outputs/handwritting_model.pth"))

model.eval()

index = random.randint(0, len(test_data[0]))
print(index,len(test_data[0]))
# index = 5
x, y = test_data[index][0], test_data[index][1]
print(x.shape)

plt.figure()
plt.imshow(x.squeeze(), cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()

with torch.no_grad():
    pred = model(x)
    print(pred)
    predicted, actual = pred[0].argmax(0),y
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
