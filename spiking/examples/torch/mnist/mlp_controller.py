import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
from collections import OrderedDict
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import yaml
import argparse
import objgraph
sys.path.append("C:\\Users\\tavar\\Desktop\\Thesis\\Code\\spiking\\core\\torch")
sys.path.append("C:\\Users\\tavar\\Desktop\\Thesis\\Code\\spiking")
from core.torch.model import get_model
from examples.torch.mnist.controller_train import ModelEncoder, ModelMasked
from examples.torch.mnist.mlp_controller_ChatGPT import NeuralNetwork


# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, csv_file, steps, labels):
        self.data = pd.read_csv(csv_file, header=0, usecols=labels)
        self.constant_columns = self.data.loc[:, :].columns[self.data.loc[:, :].std(axis=0) <= 1e-10]
        self.max_array = self.data.loc[:, :].max(axis=0)
        self.min_array = self.data.loc[:, :].min(axis=0)
        self.normalized_arr = self.data.copy()
        for col in self.data.columns:
            if col not in self.constant_columns:
                self.normalized_arr[col] = (self.data[col] - self.min_array[col]) / (self.max_array[col] - self.min_array[col])
        self.steps = steps
        self.values = self.normalized_arr.values
        self.length = int(len(self.data)/self.steps)
        self.normalized_final_array = self.values.reshape(self.length, self.steps, len(self.values[0]))      

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.item = self.normalized_final_array[idx, :]
        self.item = torch.tensor(self.item, dtype=torch.float32)
        return self.item

# Define your sequence-to-sequence RNN model
class Seq2SeqRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through RNN
        out, _ = self.rnn(x, h0)
        
        # Fully connected layer
        out = self.fc(out)
        
        return out
    
def main(spiking_input):
    # model_controller = Seq2SeqRNN(24, 120, 4)
    model_controller = NeuralNetwork()

    optimizer = torch.optim.Adam(model_controller.parameters(), lr=0.1)
    optimizer.zero_grad()

    # training params
    device = "cpu"
    device = torch.device(device)
    steps = 2000

    # Specify the path to your CSV file
    csv_file_path = 'C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_control.csv'

    # input_labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4",
    #                 "Mx", "My", "Mz", "Fz", "gate_x", "gate_y", "gate_z", "gate_yaw"]
    input_labels = ["phi", "p"]
    
    output_labels = ["u1", "u2", "u3", "u4"]
    epochs = 5
    batch_size = 4

    # Create a dataset instance
    dataset_output = MyDataset(csv_file_path, steps, output_labels) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    # dataset_output = dataset_output[0:100]

    dataset_input = MyDataset(csv_file_path, steps, input_labels) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    # dataset_input = dataset_input[0:100]
    if spiking_input:
        dataset_input_loader = DataLoader(dataset_input)
        x = next(iter(dataset_input_loader))

        block_sizes = [(6, 45), (6, 45), (4, 20), (4, 20), (4, 30)]
        output_size = (sum(block[0] for block in block_sizes), sum(block[1] for block in block_sizes))
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="encoder_train_parameters.yaml")
        parser.add_argument("--debug", action="store_true")
        args = parser.parse_args()

        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config["model"]["p1"]["out_features"] = output_size[0]
        config["model"]["e1"]["synapse"]["out_features"] = output_size[1]

        # Initialize a zero-filled mask matrix
        mask_matrix = torch.zeros(output_size)
        # Set non-zero values for blocks along the main diagonal
        current_row = 0
        current_column = 0
        for block_rows, block_cols in block_sizes:
            for i in range(block_rows):
                for j in range(block_cols):
                    row = current_row + i
                    col = current_column + j
                    if row < output_size[0] and col < output_size[1]:
                        mask_matrix[row, col] = 1
            current_row += block_rows
            current_column += block_cols

        pretrained_model = get_model(ModelMasked, config["model"], data=x[0, :], device=device, mask=mask_matrix)
        pretrained_model.load_state_dict(torch.load("C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_control_input_parameters_masked.pth"))
        pretrained_dict = pretrained_model.state_dict()

        del(config["model"]["p1"])
        model = get_model(ModelEncoder, config["model"], data=x[0, :], device=device, mask=mask_matrix)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)

        model.eval()
        input_list = torch.Tensor()

        with torch.no_grad():
            with tqdm(dataset_input_loader) as batches_loop:
                for data in batches_loop:
                    data = data.to(device)
                    data = data.swapaxes(0, 1)  # (b, t, c, h, w) -> (t, b, c, h, w)
                    save_steps = torch.Tensor()
                    for x, _ in zip(data, data):
                        yhat = model(x[0])
                        save_steps = torch.cat([save_steps, yhat.unsqueeze(0)], dim=0)
                    input_list = torch.cat([input_list, save_steps.unsqueeze(0)], dim=0)
        dataset_input = torch.FloatTensor(input_list)
        del(input_list)

    print("Dataset_size input:", len(dataset_input), "Dataset_size output:", len(dataset_output))
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        for i in range(0, len(dataset_input), batch_size):
            inputs = dataset_input[i:i+batch_size]
            targets = dataset_output[i:i+batch_size]

            # Forward pass
            outputs = model_controller(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    torch.save(model_controller.state_dict(), "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_controller_chatGPT_p_phi.pth")

if __name__ == "__main__":
    main(False)