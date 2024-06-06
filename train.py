import torch
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import time

# Custom Dataset class to handle data loading from JSON files
class StockDataset(Dataset):
    def __init__(self, input_file, label_file, chunk_size=1000):
        self.input_file = input_file
        self.label_file = label_file
        self.chunk_size = chunk_size

    def load_json_chunk(self, file, start_idx, end_idx):
        with open(file) as f:
            f.seek(start_idx)
            data = []
            while f.tell() < end_idx:
                line = f.readline()
                if not line:
                    break
                data.append(json.loads(line))
        return data

    def __len__(self):
        with open(self.input_file) as f:
            num_lines = sum(1 for line in f)
        return num_lines // self.chunk_size

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = (idx + 1) * self.chunk_size
        print(f"Loading chunk [{start_idx}:{end_idx}]")
        inputs = self.load_json_chunk(self.input_file, start_idx, end_idx)
        labels = self.load_json_chunk(self.label_file, start_idx, end_idx)
        print(f"Loaded inputs: {len(inputs)} samples, Loaded labels: {len(labels)} samples")
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


class StockPredictor(nn.Module):
    def __init__(self, num_features):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 5)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

def train_model(model, data_loader, criterion, optimizer, num_epochs, grad_accum_steps):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.cuda(), labels.cuda()  # Move to GPU

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()

            if (i + 1) % grad_accum_steps == 0:  # Gradient accumulation
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:  # Print every 100 batches
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        torch.cuda.empty_cache()  # Clear cache at the end of each epoch

def estimate_training_time(data_loader, model, criterion, optimizer, num_test_batches=100):
    model.train()
    start_time = time.time()

    for i, (inputs, labels) in enumerate(data_loader):
        if i >= num_test_batches:
            break
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    time_per_batch = (end_time - start_time) / num_test_batches
    return time_per_batch

if __name__ == "__main__":
    input_file = 'inputs.json'
    label_file = 'labels.json'
    
    batch_size = 16  # Reduce batch size to mitigate memory issues
    chunk_size = 1000  # Number of lines to read from each file per chunk
    num_features = 5  # Number of input features
    grad_accum_steps = 4  # Number of steps to accumulate gradients

    # Model
    model = StockPredictor(num_features)
    model = model.cuda()  # Move model to GPU

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Estimate training time per epoch using the first chunk
    dataset = StockDataset(input_file, label_file, chunk_size=chunk_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    time_per_batch = estimate_training_time(data_loader, model, criterion, optimizer)
    full_steps_per_epoch = len(dataset)
    estimated_time_per_epoch = time_per_batch * full_steps_per_epoch
    total_epochs = 10
    total_time = estimated_time_per_epoch * total_epochs

    print(f"Estimated time per epoch: {estimated_time_per_epoch:.2f} seconds")
    print(f"Estimated total training time for {total_epochs} epochs: {total_time / 3600:.2f} hours")

    # Train model over all chunks
    num_epochs = 10

    for epoch in range(num_epochs):
        train_model(model, data_loader, criterion, optimizer, 1, grad_accum_steps)
