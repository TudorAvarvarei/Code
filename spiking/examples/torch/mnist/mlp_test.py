import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("C:\\Users\\tavar\\Desktop\\Thesis\\Code\\spiking\\core\\torch")
sys.path.append("C:\\Users\\tavar\\Desktop\\Thesis\\Code\\spiking")
import matplotlib.pyplot as plt
from examples.torch.mnist.mlp_controller import Seq2SeqRNN
from examples.torch.mnist.mlp_controller_ChatGPT import NeuralNetwork
from core.torch.model import get_model
from examples.torch.mnist.controller_train import ModelEncoder, ModelMasked
import argparse
import yaml


class MyDataset(Dataset):
    def __init__(self, training_csv_file, testing_csv_file, steps, param):
        self.training_data = pd.read_csv(training_csv_file, header=0, usecols=param)
        self.data = pd.read_csv(testing_csv_file, header=0, usecols=param)
        self.constant_columns = self.data.loc[:, :].columns[self.data.loc[:, :].std(axis=0) <= 1e-10]
        self.max_array = self.training_data.loc[:, :].max(axis=0)
        self.min_array = self.training_data.loc[:, :].min(axis=0)
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
    

def main(spiking_input):
    # Specify the path to your CSV file
    testing_csv_file = 'C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_control_test.csv'
    training_csv_file = 'C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_control.csv'

    labels_input = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4", 
                    "Mx", "My", "Mz", "Fz", "gate_x", "gate_y", "gate_z", "gate_yaw"]
    # labels_input = ["phi", "p"]
    
    labels_output = ["u1", "u2", "u3", "u4"]

    # Create a dataset instance
    dataset_input = MyDataset(training_csv_file, testing_csv_file, 1, labels_input) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    print("Dataset_size:", len(dataset_input))

    if spiking_input:
        device = torch.device("cpu")
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

    # Create a dataset instance
    dataset_output = MyDataset(training_csv_file, testing_csv_file, 1, labels_output) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    print("Dataset_size:", len(dataset_output))
    
    # model = MLP_model()
    # model = Seq2SeqRNN(24, 500, 4)
    model_controller = NeuralNetwork()
    # model = Seq2SeqRNN(24, 120, 4)
    model_controller_path = "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_controller_chatGPT.pth"
    model_controller.load_state_dict(torch.load(model_controller_path))

    model_controller.eval()
    output_list = torch.Tensor()
    target_list = torch.Tensor()

    batch_size = 1
    error_list = []

    for i in range(0, len(dataset_input), batch_size):
        input = dataset_input[i:i + batch_size]
        estimated_output = model_controller(input)
        target = dataset_output[i:i + batch_size]
        target_list = torch.cat([target_list, target.unsqueeze(0)], dim=0)
        error = abs(estimated_output - target).cpu().detach().numpy()
        error_list.append(error[0].tolist())
        output_list = torch.cat([output_list, estimated_output.unsqueeze(0)], dim=0)

    error_list = np.array(error_list)
    error_list = error_list.reshape((10000, 4))
    averaged_error = np.mean(error_list, axis=0)
    for i, label in enumerate(labels_output):
        print(label, averaged_error[i])

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    fig.tight_layout(pad=4.0)
    fig.suptitle("Input parameters")
    output_list = torch.permute(output_list, (3, 1, 0, 2))
    output_list = torch.flatten(output_list, start_dim=1)
    target_list = torch.permute(target_list, (3, 1, 0, 2))
    target_list = torch.flatten(target_list, start_dim=1)
    for i, ax in enumerate(axes.flat):
        ax.plot(target_list[i], label='Real')
        ax.plot(output_list[i].cpu().detach().numpy(), label='Estimated')
        # ax.set_xlabel('Timestep')
        ax.grid(True)
        ax.legend()
        ax.set_title(labels_output[i])
    plt.show()

if __name__ == "__main__":
    main(False)