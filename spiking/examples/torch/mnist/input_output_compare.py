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
from examples.torch.mnist.encoder_train import Model
from core.torch.model import get_model, BaseModel
from core.torch.layer import LinearCubaLif
from core.torch.synapse import MaskedLazyLinear
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from examples.torch.mnist.encoder_train_masked import ModelMasked
from examples.torch.mnist.disturbance_encoder import encoder, decoder

class Model(BaseModel):
    def __init__(self, e1, p1):
        super().__init__()

        self.e1 = LinearCubaLif(**e1)
        self.p1 = nn.LazyLinear(**p1)

    def forward(self, input):
        x = self.e1(input)
        # spikes.append(x.tolist())
        x = x.flatten(start_dim=1)
        x = self.p1(x)
        return x

    def reset(self):
        self.e1.reset()

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

def main(config, model_path, labels):
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

    # Create a dataset instance
    dataset = MyDataset(training_csv_file, testing_csv_file, labels) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    print("Dataset_size:", len(dataset))
    
    # Create a DataLoader
    testing_dataset = DataLoader(dataset, shuffle=False)

    # # get model and trace it
    x = next(iter(testing_dataset))
    print("Input size:", x[0, :].size())
    # pretrained_model = get_model(Model, config["model"], data=x[0, :], device=device)
    # pretrained_model.load_state_dict(torch.load(model))
    # pretrained_dict = pretrained_model.state_dict()

    # model = get_model(Model, config["model"], data=x[0, :], device=device)
    # Define block sizes and positions along the main diagonal
    block_sizes = [(6, 45), (6, 45), (4, 30), (4, 20)]
    output_size = (sum(block[0] for block in block_sizes), sum(block[1] for block in block_sizes))
    config["model"]["e1"]["synapse"]["out_features"] = output_size[1]
    config["model"]["p1"]["out_features"] = output_size[0]

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
    # config["model"]["p1"]["out_features"] = len(labels)
    # pretrained_model = get_model(Model, config["model"], data=x[0, :], device=device)
    pretrained_model.load_state_dict(torch.load(model_path))
    pretrained_dict = pretrained_model.state_dict()
    model = get_model(ModelMasked, config["model"], data=x[0, :], device=device, mask=mask_matrix)
    # model = get_model(Model, config["model"], data=x[0, :], device=device)


    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)

    model.eval()
    error = []
    input = []
    output = []
    with torch.no_grad():
        with tqdm(testing_dataset) as batches_loop:
            for data in batches_loop:
                data = data.to(device)
                data = data.swapaxes(0, 1)  # (b, t, c, h, w) -> (t, b, c, h, w)
                for x, y in zip(data, data):
                    yhat = model(x)
                    error.append(abs(y-yhat).cpu().numpy())
                    input.append(x.cpu().numpy())
                    output.append(yhat.cpu().numpy())
                    # print("y_real", y)
                    # print("y_predicted", yhat)
    error = np.squeeze(np.array(error))
    input = np.squeeze(np.array(input))
    output = np.squeeze(np.array(output))
    average_error = np.average(error, axis=0)
    
    average_error_dict = dict(zip(labels, average_error))
    print(average_error_dict)

    # # Create a dataset instance
    labels_disturbances = ["Mx", "My", "Mz", "Fz"]
    dataset_disturbances = MyDataset(training_csv_file, testing_csv_file, labels_disturbances) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    no_spiking_neurons_disturbances = 20
    error_disturbances = []
    input_disturbances = []
    spikes_array_disturbances = []
    output_disturbances = []
    # Create a DataLoader
    testing_dataset_disturbances = DataLoader(dataset_disturbances, shuffle=False)
    with torch.no_grad():
        with tqdm(testing_dataset_disturbances) as batches_loop:
            for data_disturbances in batches_loop:
                data_disturbances = data_disturbances.to(device)
                data_disturbances = data_disturbances.swapaxes(0, 1)  # (b, t, c, h, w) -> (t, b, c, h, w)
                for x, y in zip(data_disturbances, data_disturbances):
                    spikes = encoder(no_spiking_neurons_disturbances, x)
                    yhat = decoder(spikes, labels_disturbances)
                    error_disturbances.append(abs(y-yhat).cpu().numpy())
                    input_disturbances.append(x.cpu().numpy())
                    spikes_array_disturbances.append(spikes.cpu(). numpy())
                    output_disturbances.append(yhat.cpu().numpy())
                    # print("y_real", y)
                    # print("y_predicted", yhat)
    error_disturbances = np.squeeze(np.array(error_disturbances))
    input_disturbances = np.squeeze(np.array(input_disturbances))
    output_disturbances = np.squeeze(np.array(output_disturbances))
    average_error_disturbances = np.average(error_disturbances, axis=0)
    
    average_error_dict_disturbances = dict(zip(labels_disturbances, average_error_disturbances))
    print(average_error_dict_disturbances)

    # # Create a dataset instance
    labels_output = ["u1", "u2", "u3", "u4"]
    dataset_output = MyDataset(training_csv_file, testing_csv_file, labels_output) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    
    # Create a DataLoader
    testing_dataset_output = DataLoader(dataset_output, shuffle=False)

    # # get model and trace it
    x = next(iter(testing_dataset_output))
    print("Input size:", x[0, :].size())
    # pretrained_model = get_model(Model, config["model"], data=x[0, :], device=device)
    # pretrained_model.load_state_dict(torch.load(model))
    # pretrained_dict = pretrained_model.state_dict()
    
    # pretrained_model = get_model(ModelMasked, config["model"], data=x[0, :], device=device, mask=mask_matrix)
    model_output = "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_fix_output_refined_30_neurons.pth"
    config["model"]["e1"]["synapse"]["out_features"]=30
    config["model"]["p1"]["out_features"]=4
    pretrained_model_output = get_model(Model, config["model"], data=x[0, :], device=device)
    pretrained_model_output.load_state_dict(torch.load(model_output))
    pretrained_dict_output = pretrained_model_output.state_dict()
    model_output = pretrained_model_output

    model_output.eval()
    error_output = []
    input_output = []
    output_output = []
    with torch.no_grad():
        with tqdm(testing_dataset_output) as batches_loop:
            for data_output in batches_loop:
                data_output = data_output.to(device)
                data_output = data_output.swapaxes(0, 1)  # (b, t, c, h, w) -> (t, b, c, h, w)
                for x, y in zip(data_output, data_output):
                    yhat = model_output(x)
                    error_output.append(abs(y-yhat).cpu().numpy())
                    input_output.append(x.cpu().numpy())
                    output_output.append(yhat.cpu().numpy())
                    # print("y_real", y)
                    # print("y_predicted", yhat)
    error_output = np.squeeze(np.array(error_output))
    input_output = np.squeeze(np.array(input_output))
    output_output = np.squeeze(np.array(output_output))
    average_error_output = np.average(error_output, axis=0)
    
    average_error_dict_output = dict(zip(labels_output, average_error_output))
    print(average_error_dict_output)

    at_once = {'x': 0.055736408, 'y': 0.044692323, 'z': 0.053857062, 'vx': 0.05906153, 'vy': 0.065190874, 'vz': 0.0408006, 
            'phi': 0.04794844, 'theta': 0.041651208, 'psi': 0.075179264, 'p': 0.048515547, 'q': 0.035467044, 'r': 0.053818513, 
            'w1': 0.054159813, 'w2': 0.044427726, 'w3': 0.05422948, 'w4': 0.060256585, 'Mx': 0.07205693, 'My': 0.11504696, 'Mz': 0.07644945, 
            'Fz': 0.069524504, 'gate_x': 0.0385971, 'gate_y': 0.02322253, 'gate_z': 0.036343813, 'gate_yaw': 0.021013513, 
            'u1': 0.062029053, 'u2': 0.05029848, 'u3': 0.06452216, 'u4': 0.06083837}

    batches = {'x': 0.02790151, 'y': 0.025045095, 'z': 0.023679191, 'vx': 0.029735671, 'vy': 0.027135199, 'vz': 0.024110133,
            'phi': 0.024214504, 'theta': 0.023538098, 'psi': 0.027750256, 'p': 0.027427524, 'q': 0.024677847, 'r': 0.024158109,
            'w1': 0.043472435, 'w2': 0.031815764, 'w3': 0.035546567, 'w4': 0.03258463,
            'Mx': 0.010618235603455158, 'My': 0.011128867537744527, 'Mz': 0.009189414978026882, 'Fz': 0.008842261952739221,
            'gate_x': 0.00018011351, 'gate_y': 0.00014677424, 'gate_z': 6.4841784e-05, 'gate_yaw': 0.0001055058, 
            'u1': 0.0835511, 'u2': 0.06660617, 'u3': 0.06329836, 'u4': 0.07227252}

    # fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 12))
    # fig.tight_layout(pad=4.0)
    labels = ["Mx", "My", "Mz", "Fz"]
    fig = plt.figure(figsize=(10, 12))  # Adjust figsize as needed
    gs = GridSpec(len(labels), 2, figure=fig, width_ratios=[4, 1], hspace=0.4)
    fig.tight_layout(pad=4.0)
    fig.suptitle("Autoencoder output performance for Distrubances", fontsize=18)
    # Add common x-label to the bottom row of left plots
    # fig.text(0.5, 0.08, 'Timestep', ha='center', fontsize=14)

    # Add common y-label to the middle of the left and right plots
    fig.text(0.03, 0.5, 'Output', va='center', rotation='vertical', fontsize=14)
    
    # Add common y-label to the middle of the left and right plots
    fig.text(0.73, 0.5, 'MSE_Loss', va='center', rotation='vertical', fontsize=14)
    for i in range(len(labels)):
        ax_main = fig.add_subplot(gs[i, 0])
        ax_main.plot(input_disturbances[:, i], label='Real')
        ax_main.plot(output_disturbances[:, i], label='Estimated')
        ax_main.grid(True)
        ax_main.set_title(labels[i])
        if i == len(labels) - 1:
            ax_main.set_xlabel('Timestep', fontdict={"size":14})
        # ax_main.set_ylabel('Output')
        # ax_main.legend(loc="best")

        ax_bar = fig.add_subplot(gs[i, 1])
        ax_bar.bar("All at once", list(at_once.values())[i + 16], width=0.5, color="grey")
        ax_bar.bar("Batches", list(batches.values())[i + 16], width=0.5, color="C1")
        ax_bar.grid(True)
        ax_bar.set_title(labels[i])
        # ax_bar.set_ylabel("MSE_Loss")
        # ax.legend(loc='best', fontsize=12)
        # ax_bar.set_yticks([])

    # fig.suptitle("Input parameters")
    # for i, ax in enumerate(axes.flat):
    #     ax.plot(input[:, i], label='Real')
    #     ax.plot(output[:, i], label='Estimated')
    #     ax.grid(True)
    #     ax.set_title(labels[i])
        # if (i < len(labels)):
        #     ax.plot(input[:, i], label='Real')
        #     ax.plot(output[:, i], label='Estimated')
        #     ax.grid(True)
        #     ax.set_title(labels[i])
        # elif (i < len(labels) + len(labels_disturbances)):
        #     ax.plot(input_disturbances[:, i - len(labels)], label='Real')
        #     ax.plot(output_disturbances[:, i - len(labels)], label='Estimated')
        #     ax.grid(True)
        #     ax.set_title(labels_disturbances[i - len(labels)])
        # elif (i < len(labels) + len(labels_disturbances) + len(labels_output)):
        #     ax.plot(input_output[:, i - len(labels) - len(labels_disturbances)], label='Real')
        #     ax.plot(output_output[:, i - len(labels) - len(labels_disturbances)], label='Estimated')
        #     ax.grid(True)
        #     ax.set_title(labels_output[i - len(labels) - len(labels_disturbances)])
    plt.show()

    return average_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="encoder_train_parameters.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # models = ["C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_50_neurons_refined.pth",
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_75_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_100_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_150_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_200_neurons.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_300_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_400_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_500_neurons_refined.pth"]
        
    model = "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_control_input_parameters_masked_no_disturbances.pth"
    
    # neurons = [50, 75, 100, 150, 200, 300, 400, 500]
    # average_errors = []
    # labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4", 
    #           "Mx", "My", "Mz", "Fz", "gate_x", "gate_y", "gate_z", "gate_yaw", "u1", "u2", "u3", "u4"]
    
    labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4", 
              "gate_x", "gate_y", "gate_z", "gate_yaw"]

    _ = main(config, model, labels)
    
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
