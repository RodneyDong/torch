import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os


column = 6
window = 30

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        global window,columns,batch_global
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(column*window, window),
            nn.ReLU(),  # Rectified Linear Unit
            nn.Linear(window, column),
            nn.ReLU(),
            nn.Linear(column, 2)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
        # Extract inputs starting from the 3rd column, and group them into 30 groups of 6 elements
        self.inputs = self.data.iloc[:, 2:].values.reshape(-1, 30, 6)  # Reshape to 30x6 groups
        
        # Convert the first two columns to binary labels
        self.outputs = self.data.iloc[:, :2].idxmax(axis=1).astype(int).values  # Convert to integers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.outputs[idx], dtype=torch.long)  # CrossEntropyLoss expects long tensor for targets
        return X, y


# Define the training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print("Batches: ", batch)

if __name__ == '__main__':
    # Path to the CSV file generated from the previous program
    csv_file = 'stockdata/SPY_TrainingData_30_13.csv'

    # Load the dataset
    dataset = CustomDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Get the device for training
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    # Initialize the model, loss function, and optimizer
    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train the model
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(dataloader, model, loss_fn, optimizer)
    print("Done with training!")

    # Save the trained model
    filepath = os.path.join("outputs", "stockTestModel1.pth")
    torch.save(model.state_dict(), filepath)
    print("Saved PyTorch Model State to stockTestModel1.pth")