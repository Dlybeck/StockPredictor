# step 1: importing libraries  
import torch 
import torch.nn as nn
import json

class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_dict):
        x = input_dict['sequence_data']
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out

# Example usage
input_size = 10
hidden_size = 20
output_size = 1
model = SequenceModel(input_size, hidden_size, output_size)

# Example input dictionary with a batch of sequences
sequence_data = torch.randn(5, 15, input_size)  # Batch size of 5, sequence length of 15
input_dict = {'sequence_data': sequence_data}

# Pass dictionary to the model
output = model(input_dict)
print(output)
    
    
def load_data_and_labels():
    with open('sp500.json', 'r') as f:
        data = json.load(f)
    
    training_data = []
    labels = []
    
    #for every company
    for comp in data:
        #for every day
        for day in comp:
            for k, v in day.items:
                if(k != "Change") 
        