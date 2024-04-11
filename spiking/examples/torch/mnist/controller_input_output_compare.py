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


class ModelDecoder(BaseModel):
    def __init__(self, p1):
        super().__init__()
        self.p1 = nn.LazyLinear(**p1)

    def forward(self, input):
        x = self.p1(input)
        return x

    def reset(self):
        pass

class ModelController(BaseModel):
    def __init__(self, e1, e2, e3):
        super().__init__()

        self.e1 = LinearCubaLif(**e1)
        self.e2 = LinearCubaLif(**e2)
        self.e3 = LinearCubaLif(**e3)

    def forward(self, input):
        x = self.e1(input)
        x = x.flatten(start_dim=1)
        x = self.e2(x)
        x = x.flatten(start_dim=1)
        x = self.e3(x)
        return x

    def reset(self):
        self.e1.reset()
        self.e2.reset()
        self.e3.reset()

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

def main(config_encoder, config_decoder, config_controller, model_encoder_path, model_decoder_path, model_controller_path):
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
    # Initialize a zero-filled mask matrix
    block_sizes = [(6, 45), (6, 45), (4, 20), (4, 20), (4, 30)]
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
                    yhat = model_encoder(x_encoder[0])
                    save_steps = torch.cat([save_steps, yhat.unsqueeze(0)], dim=0)
                input_list = torch.cat([input_list, save_steps.unsqueeze(0)], dim=0)
    input_spiking_dataset = torch.FloatTensor(input_list)
    del(input_list)
    
    x_controller = next(iter(input_spiking_dataset))
    print("Input size:", x_controller[0, :].size())
    # pretrained_model = get_model(ModelMasked, config["model"], data=x[0, :], device=device, mask=mask_matrix)
    pretrained_model_controller = get_model(ModelController, config_controller["model"], data=x_controller[0, :], device=device)
    pretrained_model_controller.load_state_dict(torch.load(model_controller_path))
    pretrained_dict_controller = pretrained_model_controller.state_dict()
    # model = get_model(ModelMasked, config["model"], data=x[0, :], device=device, mask=mask_matrix)
    model_controller = get_model(ModelController, config_controller["model"], data=x_controller[0, :], device=device)
    model_dict_controller = model_controller.state_dict()

    pretrained_dict_controller = {k: v for k, v in pretrained_dict_controller.items() if k in model_dict_controller}
    model_dict_controller.update(pretrained_dict_controller)
    model_controller.load_state_dict(pretrained_dict_controller)
    model_controller.eval()

    output_list = torch.Tensor()
    with torch.no_grad():
        with tqdm(input_spiking_dataset) as batches_loop:
            for data in batches_loop:
                data = data.to(device)
                save_steps = torch.Tensor()
                for x, y in zip(data, data):
                    yhat = model_controller(x)
                    save_steps = torch.cat([save_steps, yhat.unsqueeze(0)], dim=0)
                output_list = torch.cat([output_list, save_steps.unsqueeze(0)], dim=0)
    output_spiking_dataset = torch.FloatTensor(output_list)
    del(output_list)

    # Create a dataset instance
    dataset_output = MyDataset(training_csv_file, testing_csv_file, labels_output) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    print("Dataset_size:", len(dataset_output))
    
    # Create a DataLoader
    testing_dataset_output = DataLoader(dataset_output, shuffle=False)
    # get model and trace it
    x_decoder = next(iter(testing_dataset_output))
    config_decoder["model"]["e1"]["synapse"]["out_features"] = 30
    config_decoder["model"]["p1"]["out_features"] = 4
    pretrained_model_decoder = get_model(Model, config_decoder["model"], data=x_decoder[0, :], device=device)
    pretrained_model_decoder.load_state_dict(torch.load(model_decoder_path))
    pretrained_dict_decoder = pretrained_model_decoder.state_dict()

    x_decoder = next(iter(output_spiking_dataset))
    print("Input size:", x_decoder[0, :].size())
    del(config_decoder["model"]["e1"])
    model_decoder = get_model(ModelDecoder, config_decoder["model"], data=x_decoder[0, :], device=device)
    model_dict_decoder = model_decoder.state_dict()

    pretrained_dict_decoder = {k: v for k, v in pretrained_dict_decoder.items() if k in model_dict_decoder}
    model_dict_decoder.update(pretrained_dict_decoder)
    model_decoder.load_state_dict(pretrained_dict_decoder)

    model_decoder.eval()

    combined_dataloader = zip(output_spiking_dataset, testing_dataset_output)

    error = []
    real_output = []
    estimated_output = []
    with torch.no_grad():
        with tqdm(combined_dataloader) as batches_loop:
            for data, target in batches_loop:
                data = data.to(device)
                target = target.to(device)
                for x, y in zip(data, target):
                    yhat = model_decoder(x)
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
    fig.suptitle("Input parameters")
    for i, ax in enumerate(axes.flat):
        ax.plot(real_output[:, i], label='Real')
        ax.plot(estimated_output[:, i], label='Estimated')
        # ax.set_xlabel('Timestep')
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

    parser_decoder = argparse.ArgumentParser()
    parser_decoder.add_argument("--config", type=str, default="encoder_train_parameters.yaml")
    parser_decoder.add_argument("--debug", action="store_true")
    args_decoder = parser_decoder.parse_args()

    with open(args_decoder.config, "r") as f:
        config_decoder = yaml.load(f, Loader=yaml.FullLoader)

    parser_controller = argparse.ArgumentParser()
    parser_controller.add_argument("--config", type=str, default="controller_train.yaml")
    parser_controller.add_argument("--debug", action="store_true")
    args_controller = parser_controller.parse_args()

    with open(args_controller.config, "r") as f:
        config_controller = yaml.load(f, Loader=yaml.FullLoader)

    # models = ["C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_50_neurons_refned.pth",
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_75_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_100_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_150_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_200_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_300_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_400_neurons_refined.pth", 
    #           "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_500_neurons_refined.pth"]
        
    model_encoder = "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_control_input_parameters_masked.pth"
    model_decoder = "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_control_output_neurons_30.pth"
    model_controller = "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_controller_full_dataset.pth"
    
    # neurons = [50, 75, 100, 150, 200, 300, 400, 500]
    # average_errors = []
    # labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4", 
    #           "Mx", "My", "Mz", "Fz", "gate_x", "gate_y", "gate_z", "gate_yaw", "u1", "u2", "u3", "u4"]
    
    # labels = ["gate_x", "gate_y", "gate_z", "gate_yaw"]

    _ = main(config_encoder, config_decoder, config_controller, model_encoder, model_decoder, model_controller)
    
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
