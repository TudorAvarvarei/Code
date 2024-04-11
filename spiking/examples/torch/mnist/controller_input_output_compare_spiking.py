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
from examples.torch.mnist.disturbance_encoder import encoder

class ModelEncoder(BaseModel):
    def __init__(self, e1, mask=None):
        super().__init__()
        if mask is not None:
            self.e1 = MaskedLinearCubaLif(**e1, mask=torch.t(mask))
        else:
            self.e1 = LinearCubaLif(**e1)

    def forward(self, input):
        x = self.e1(input)
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
        # self.steps = steps
        # self.values = self.normalized_arr.values
        # self.length = int(len(self.data)/self.steps)
        # self.normalized_final_array = self.values.reshape(self.length, self.steps, len(self.values[0]))

    def __len__(self):
        return len(self.normalized_arr_reduced)
        # return self.length

    def __getitem__(self, idx):
        self.item = self.normalized_arr_reduced.iloc[idx, :]
        # self.item = self.normalized_final_array[idx, :]
        self.item = torch.tensor(self.item, dtype=torch.float32)
        return self.item.unsqueeze(0)
        # return self.item

def main(config_encoder, config_controller, model_encoder_path, model_controller_path, spiking_input=False, separate_disturbances=False):
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
    
    labels_output = ["u1", "u2", "u3", "u4"]

    # Create a dataset instance
    dataset_input = MyDataset(training_csv_file, testing_csv_file, labels_input) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    print("Dataset_size:", len(dataset_input))
    
    # Create a DataLoader
    testing_dataset_input = DataLoader(dataset_input, shuffle=False)
    # testing_dataset_input = dataset_input[:-1]

    if spiking_input:
        if separate_disturbances:
            labels_input = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4",
                            "gate_x", "gate_y", "gate_z", "gate_yaw"]
            dataset_input = MyDataset(training_csv_file, testing_csv_file, labels_input) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
            block_sizes = [(6, 45), (6, 45), (4, 30), (4, 20)]
            testing_dataset_input = DataLoader(dataset_input, shuffle=False)
        else:
            block_sizes = [(6, 45), (6, 45), (4, 30), (4, 20), (4, 20)]
        # Initialize a zero-filled mask matrix
        output_size = (sum(block[0] for block in block_sizes), sum(block[1] for block in block_sizes))
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

        # get model and trace it
        x_encoder = next(iter(testing_dataset_input))
        print("Input size:", x_encoder[0, :].size())
        config_encoder["model"]["e1"]["synapse"]["out_features"] = output_size[1]
        config_encoder["model"]["p1"]["out_features"] = output_size[0]
        pretrained_model_encoder = get_model(ModelMasked, config_encoder["model"], data=x_encoder[0, :], device=device, mask=mask_matrix)
        pretrained_model_encoder.load_state_dict(torch.load(model_encoder_path))
        pretrained_dict_encoder = pretrained_model_encoder.state_dict()

        del(config_encoder["model"]["p1"])
        model_encoder = get_model(ModelEncoder, config_encoder["model"], data=x_encoder[0, :], device=device, mask=mask_matrix)
        model_dict_encoder = model_encoder.state_dict()

        pretrained_dict_encoder = {k: v for k, v in pretrained_dict_encoder.items() if k in model_dict_encoder}
        model_dict_encoder.update(pretrained_dict_encoder)
        model_encoder.load_state_dict(pretrained_dict_encoder)

        model_encoder.eval()
        input_list = torch.Tensor()

        with torch.no_grad():
            with tqdm(testing_dataset_input) as batches_loop:
                for data in batches_loop:
                    data = data.to(device)
                    data = data.swapaxes(0, 1)  # (b, t, c, h, w) -> (t, b, c, h, w)
                    save_steps = torch.Tensor()
                    for x, _ in zip(data, data):
                        yhat = model_encoder(x[0])
                        save_steps = torch.cat([save_steps, yhat.unsqueeze(0)], dim=0)
                    input_list = torch.cat([input_list, save_steps.unsqueeze(0)], dim=0)
        input_spiking_dataset = torch.FloatTensor(input_list)
        testing_dataset_input = input_spiking_dataset
        testing_dataset_input = torch.reshape(testing_dataset_input, (len(dataset_input), 1, 1, output_size[1]))
        del(input_list, input_spiking_dataset)

        if separate_disturbances:
            labels_disturbances = ["Mx", "My", "Mz", "Fz"]
            dataset_disturbances = MyDataset(training_csv_file, testing_csv_file, labels_disturbances) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
            # dataset_disturbances = dataset_disturbances[0:20]
            no_spiking_neurons_disturbances = 20
            spikes_array_disturbances = []
            # Create a DataLoader
            testing_dataset_disturbances = DataLoader(dataset_disturbances, shuffle=False)
            with torch.no_grad():
                with tqdm(testing_dataset_disturbances) as batches_loop:
                    for data_disturbances in batches_loop:
                        data_disturbances = data_disturbances.to(device)
                        data_disturbances = data_disturbances.swapaxes(0, 1)  # (b, t, c, h, w) -> (t, b, c, h, w)
                        for x, y in zip(data_disturbances, data_disturbances):
                            spikes = encoder(no_spiking_neurons_disturbances, x)
                            spikes_array_disturbances.append(spikes.cpu().numpy())
            spikes_array_disturbances = np.array(spikes_array_disturbances)
            spikes_array_disturbances = spikes_array_disturbances.reshape(len(dataset_disturbances), 1, 1, no_spiking_neurons_disturbances)
            testing_dataset_input = torch.cat((testing_dataset_input, torch.from_numpy(spikes_array_disturbances)), dim=3)
            testing_dataset_input = testing_dataset_input.float()
    
    x_controller = next(iter(testing_dataset_input))
    print("Input size:", x_controller[0, :].size())
    model_controller = get_model(ModelController, config_controller["model"], data=x_controller[0, :], device=device)
    model_controller.load_state_dict(torch.load(model_controller_path))
    model_controller.eval()
    
    # torch.reshape(testing_dataset_input, (2000, 5, 160))
    # input_loader = DataLoader(testing_dataset_input, shuffle=True, batch_size=4, num_workers=8, pin_memory=False, drop_last=True)

    output_list = torch.Tensor()
    with torch.no_grad():
        with tqdm(testing_dataset_input) as batches_loop:
            for data in batches_loop:
                data = data.to(device)
                data = data.swapaxes(0, 1)  # (b, t, c, h, w) -> (t, b, c, h, w)
                save_steps = torch.Tensor()
                for x, _ in zip(data, data):
                    yhat = model_controller(x)
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

    # for name, param in pretrained_model_controller.named_parameters():
    #     print(name, ": ", param)

    # output_spiking_dataset = torch.squeeze(output_spiking_dataset)
    # fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(20, 20))
    # fig.tight_layout(pad=4.0)
    # fig.suptitle("Output spikes")
    # for i, ax in enumerate(axes.flat):
    #     ax.plot(output_spiking_dataset[:, i].cpu().numpy(), label='Real')
    #     ax.set_title("Neuron " + str(i + 1))
    # plt.show()

    return average_error


if __name__ == "__main__":
    parser_encoder = argparse.ArgumentParser()
    parser_encoder.add_argument("--config", type=str, default="encoder_train_parameters.yaml")
    parser_encoder.add_argument("--debug", action="store_true")
    args_encoder = parser_encoder.parse_args()

    with open(args_encoder.config, "r") as f:
        config_encoder = yaml.load(f, Loader=yaml.FullLoader)

    parser_controller = argparse.ArgumentParser()
    parser_controller.add_argument("--config", type=str, default="controller_e2e_train.yaml")
    parser_controller.add_argument("--debug", action="store_true")
    args_controller = parser_controller.parse_args()

    with open(args_controller.config, "r") as f:
        config_controller = yaml.load(f, Loader=yaml.FullLoader)
 
    # model_encoder = "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_control_input_parameters_masked_no_disturbances.pth"
    # model_controller = "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_controller_e2e_spiking_input_separate_disturbances.pth"

    model_encoder = "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_control_input_parameters_masked.pth"
    model_controller = "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_controller_e2e.pth"
    
    # neurons = [50, 75, 100, 150, 200, 300, 400, 500]
    # average_errors = []
    # labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4", 
    #           "Mx", "My", "Mz", "Fz", "gate_x", "gate_y", "gate_z", "gate_yaw", "u1", "u2", "u3", "u4"]
    
    # labels = ["gate_x", "gate_y", "gate_z", "gate_yaw"]

    _ = main(config_encoder, config_controller, model_encoder, model_controller, spiking_input=False, separate_disturbances=False)
    
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
