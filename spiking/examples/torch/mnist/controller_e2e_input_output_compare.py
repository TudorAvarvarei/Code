import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import sys
sys.path.append("C:\\Users\\tavar\\Desktop\\Thesis\\Code\\spiking\\core\\torch")
sys.path.append("C:\\Users\\tavar\\Desktop\\Thesis\\Code\\spiking")
from examples.torch.mnist.encoder_train_parameters import Model
from core.torch.model import get_model, BaseModel
from core.torch.layer import LinearCubaLif, MaskedLinearCubaLif
import matplotlib.pyplot as plt
from examples.torch.mnist.encoder_train_masked import ModelMasked
from examples.torch.mnist.controller_e2e_train_simplified import ModelController


# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, training_csv_file, testing_csv_file, param):
        self.training_data = pd.read_csv(training_csv_file, header=0, usecols=param)
        self.data = pd.read_csv(testing_csv_file, header=0, usecols=param)
        self.constant_columns = self.data.loc[:, :].columns[self.data.loc[:, :].std(axis=0) <= 1e-10]
        self.max_array = self.training_data.loc[:, :].max(axis=0)
        self.min_array = self.training_data.loc[:, :].min(axis=0)
        self.normalized_arr = self.data.copy()
        for col in self.data.columns:
            if col not in self.constant_columns:
                self.normalized_arr[col] = (self.data[col] - self.min_array[col]) / (self.max_array[col] - self.min_array[col])
        self.normalized_arr_reduced = self.normalized_arr.iloc[:, :]

    def __len__(self):
        return len(self.normalized_arr_reduced)

    def __getitem__(self, idx):
        self.item = self.normalized_arr_reduced.iloc[idx, :]
        self.item = torch.tensor(self.item, dtype=torch.float32)
        return self.item.unsqueeze(0)

def main(config, model_path):
    # Get device
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     print("Device = CUDA")
    # else:
    #     device = "cpu"
    #     print("Device = CPU")

    device = torch.device("cpu")

    # Specify the path to your CSV file
    testing_csv_file = 'C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_control_test.csv'
    training_csv_file = 'C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_control.csv'

    labels_input = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4", 
                    "Mx", "My", "Mz", "Fz", "gate_x", "gate_y", "gate_z", "gate_yaw"]
    
    # labels_input = ["phi", "p"]
    
    labels_output = ["u1", "u2", "u3", "u4"]

    # block_sizes = [(6, 45), (6, 45), (4, 20), (4, 20), (4, 30)]
    # output_size = (sum(block[0] for block in block_sizes), sum(block[1] for block in block_sizes))

    # # Initialize a zero-filled mask matrix
    # mask_input = torch.zeros(output_size)
    # # Set non-zero values for blocks along the main diagonal
    # current_row = 0
    # current_column = 0
    # for block_rows, block_cols in block_sizes:
    #     for i in range(block_rows):
    #         for j in range(block_cols):
    #             row = current_row + i
    #             col = current_column + j
    #             if row < output_size[0] and col < output_size[1]:
    #                 mask_input[row, col] = 1
    #     current_row += block_rows
    #     current_column += block_cols

    # mask_output = torch.zeros(block_sizes[-1])
    # block_rows, block_cols = block_sizes[-1]
    # for i in range(block_rows):
    #     for j in range(block_cols):
    #         row = current_row + i
    #         col = current_column + j
    #         if row < output_size[0] and col < output_size[1]:
    #             mask_output[row, col] = 1

    # mask_matrices = (mask_input, mask_output)

    # Create a dataset instance
    dataset_input = MyDataset(training_csv_file, testing_csv_file, labels_input) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    print("Dataset_size:", len(dataset_input))
    
    # Create a DataLoader
    testing_dataset_input = DataLoader(dataset_input, shuffle=False)

    # get model and trace it
    x = next(iter(testing_dataset_input))
    print("Input size:", x[0, :].size())
    model = get_model(ModelController, config["model"], data=x[0, :], device=device)#, mask=mask_matrices)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    output_list = torch.Tensor()

    with torch.no_grad():
        with tqdm(testing_dataset_input) as batches_loop:
            for data in batches_loop:
                data = data.to(device)
                data = data.swapaxes(0, 1)  # (b, t, c, h, w) -> (t, b, c, h, w)
                save_steps = torch.Tensor()
                for x, _ in zip(data, data):
                    yhat = model(x)
                    save_steps = torch.cat([save_steps, yhat.unsqueeze(0)], dim=0)
                output_list = torch.cat([output_list, save_steps.unsqueeze(0)], dim=0)
    output_list = torch.FloatTensor(output_list)

    # Create a dataset instance
    dataset_output = MyDataset(training_csv_file, testing_csv_file, labels_output) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    print("Dataset_size:", len(dataset_output))

    combined_dataloader = zip(output_list, dataset_output)

    error = []
    real_output = []
    estimated_output = []
    with torch.no_grad():
        with tqdm(combined_dataloader) as batches_loop:
            for data, target in batches_loop:
                data = data.to(device)
                target = target.to(device)
                for yhat, y in zip(data, target):
                    error.append(abs(y-yhat).cpu().numpy())
                    real_output.append(y.cpu().numpy())
                    estimated_output.append(yhat.cpu().numpy())
                    # print("y_real", y)
                    # print("y_predicted", yhat)
    error = np.squeeze(np.array(error))
    real_output = np.squeeze(np.array(real_output))
    estimated_output = np.squeeze(np.array(estimated_output))
    average_error = np.average(error, axis=0)
    
    average_error_dict = dict(zip(labels_output, average_error))
    print(average_error_dict)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    fig.tight_layout(pad=4.0)
    fig.suptitle("Input parameters", fontsize=18)
    for i, ax in enumerate(axes.flat):
        ax.plot(real_output[:, i], label='Real')
        ax.plot(estimated_output[:, i], label='Estimated')
        ax.set_xlabel('Timestep', fontdict={"size":14})
        ax.set_ylabel('Output', fontdict={"size":14})
        ax.grid(True)
        ax.set_title(labels_output[i])

    plt.show()

    return average_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="controller_e2e_train.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # models = ["C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_50_neurons_refned.pth",
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_75_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_100_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_150_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_200_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_300_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_400_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_500_neurons_refined.pth"]
    
    model = "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_controller_e2e.pth"
    

    _ = main(config, model)
    
    # for i in range(len(neurons)):
    #     config["model"]["e1"]["synapse"]["out_features"] = neurons[i]
    #     average_errors.append(main(config, models[i], labels).tolist())
    # fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(20, 20))
    # fig.tight_layout(pad=4.0)
    # for i, ax in enumerate(axes.flat):
    #     ax.plot(neurons, [column[i] for column in average_errors])
    #     ax.set_xlabel('Spiking neurons')
    #     ax.set_ylabel('RMSE error')
    #     ax.grid(True)
    #     ax.set_title(labels[i])
    # plt.show()
