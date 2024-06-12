import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Function to print current time for better tracking
def current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

print(f"{current_time()}: Loading data...")

# Load the data
with open('training_inputs.json', 'r') as f:
    inputs = json.load(f)

# Convert the input data to a NumPy array
def convert_to_array(data):
    numeric_data = []
    for sample in data:
        numeric_sample = []
        for entry in sample:
            numeric_sample.append([entry['Open'], entry['High'], entry['Low'], entry['Close'], entry['Adj Close'], entry['Volume']])
        numeric_data.append(numeric_sample)
    return np.array(numeric_data)

inputs = convert_to_array(inputs)

# Load the labels directly from the JSON file
with open('training_labels.json', 'r') as f:
    labels = np.array(json.load(f))

print(f"{current_time()}: Data loaded. Normalizing data...")

# Normalize the data
scaler = MinMaxScaler()
inputs = inputs.reshape(-1, inputs.shape[-1])
inputs = scaler.fit_transform(inputs)
inputs = inputs.reshape(-1, 100, inputs.shape[-1])  # 100 time steps

print(f"{current_time()}: Splitting data into training and testing sets...")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

print(f"{current_time()}: Converting data to PyTorch tensors...")

# Convert data to PyTorch tensors
X_train = torch.from_numpy(X_train).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

# Create PyTorch datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create PyTorch data loaders
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

print(f"{current_time()}: Defining the LSTM model...")

# Define the improved LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=50, output_size=4, num_layers=3, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel().to(device)

print(f"{current_time()}: Compiling the model...")
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

print(f"{current_time()}: Starting training...")

# Training loop
train_losses = []
test_losses = []
for epoch in range(100):
    start_time = time.time()  # Record start time for the epoch
    train_loss = 0.0
    test_loss = 0.0

    # Training loop
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation loop
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    end_time = time.time()  # Record end time for the epoch
    epoch_time = end_time - start_time  # Calculate time taken for the epoch

    print(f"Epoch {epoch+1}/{100}, Train Loss: {train_loss:.4f}, Val Loss: {test_loss:.4f}, Time: {epoch_time:.2f}s")

print(f"{current_time()}: Training completed. Saving the model...")

# Save the model
torch.save(model.state_dict(), 'stock_lstm_model.pth')

print(f"{current_time()}: Model saved. Plotting training history...")

# Plot training history
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(f"{current_time()}: Training history plotted.")