import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import yaml
import objgraph

sys.path.append("C:\\Users\\tavar\\Desktop\\Thesis\\Code\\spiking\\core\\torch")
sys.path.append("C:\\Users\\tavar\\Desktop\\Thesis\\Code\\spiking")
from core.torch.layer import LinearCubaLif, MaskedLinearCubaLif
from core.torch.model import get_model, BaseModel
from core.torch.synapse import MaskedLazyLinear
from examples.torch.mnist.controller_train import ModelEncoder, ModelMasked
from disturbance_encoder import encoder
import wandb

class ModelController(BaseModel):
    def __init__(self, e1, e2, e3, p1):
        super().__init__()

        self.e1 = LinearCubaLif(**e1)
        self.e2 = LinearCubaLif(**e2)
        self.e3 = LinearCubaLif(**e3)
        self.p1 = nn.LazyLinear(**p1)

    def forward(self, input):
        x = self.e1(input)
        x = x.flatten(start_dim=1)
        x = self.e2(x)
        x = x.flatten(start_dim=1)
        x = self.e3(x)
        x = x.flatten(start_dim=1)
        x = self.p1(x)
        return x

    def reset(self):
        self.e1.reset()
        self.e2.reset()
        self.e3.reset()

def sequence(model, data, target):
    data = data.swapaxes(0, 1)  # (b, t, c, h, w) -> (t, b, c, h, w)
    target = target.swapaxes(0, 1)

    model.reset()
    loss = 0
    
    for x, y in zip(data, target):
        yhat = model(x)
        loss = loss + F.mse_loss(yhat, y)

    loss = loss / len(data)
    # accuracy = yhat.eq(y).float().mean()
    return loss #, accuracy


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    # wandb.watch(model, criterion=sequence, log="parameters", log_freq=1)

    epoch_loss = 0
    # epoch_accuracy = 0
    passes = 0

    with tqdm(dataloader, desc="train batches", leave=False, dynamic_ncols=True) as batches_loop:
        for data, target in batches_loop:
            data, target = data.to(device), target.to(device)

            loss = sequence(model, data, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # wandb.log({"Loss":loss.item()})

            epoch_loss += loss.item()
            # epoch_accuracy += accuracy.item()
            passes += 1

            batches_loop.set_postfix_str(
                f"train loss: {epoch_loss / passes:.4f}" #, train accuracy: {epoch_accuracy / passes * 100:.2f}"
            )

    return epoch_loss / passes #, epoch_accuracy / passes

def eval_model(model, dataloader, device):
    model.eval()

    epoch_loss = 0
    # epoch_accuracy = 0
    passes = 0

    with torch.no_grad():
        with tqdm(dataloader, desc="test batches", leave=False, dynamic_ncols=True) as batches_loop:
            for data, target in batches_loop:
                data, target = data.to(device), target.to(device)

                loss = sequence(model, data, target)

                epoch_loss += loss.item()
                # epoch_accuracy += accuracy.item()
                passes += 1

                batches_loop.set_postfix_str(
                    f"test loss: {epoch_loss / passes:.4f}" #, test accuracy: {epoch_accuracy / passes * 100:.2f}"
                )

    return epoch_loss / len(dataloader) #, epoch_accuracy / len(dataloader)
    
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
    
class DatasetController(Dataset):
    def __init__(self, input, output):
        self.input=input
        self.output=output
        self.length=len(input)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.input_item = self.input[idx]
        self.output_item = self.output[idx]
        # self.input_item = torch.tensor(self.input_item, dtype=torch.float32)
        # self.output_item = torch.tensor(self.output_item, dtype=torch.float32)
        return self.input_item, self.output_item


def main(config, args, spiking_input=True, separate_disturbances=True):
    # training params
    device = "cpu"
    device = torch.device(device)
    steps = config["dataset"]["steps"]

    # Specify the path to your CSV file
    csv_file_path = 'C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_control.csv'

    input_labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4", "Mx", "My", "Mz", "Fz",
                    "gate_x", "gate_y", "gate_z", "gate_yaw"]
    
    # input_labels = ["phi", "p"]
    
    output_labels = ["u1", "u2", "u3", "u4"]

    # Create a dataset instance
    dataset_output = MyDataset(csv_file_path, steps, output_labels) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    dataset_output = dataset_output[0:20]

    dataset_input = MyDataset(csv_file_path, steps, input_labels) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    dataset_input = dataset_input[0:20]

    if spiking_input:
        if separate_disturbances:
            input_labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4",
                            "gate_x", "gate_y", "gate_z", "gate_yaw"]
            dataset_input = MyDataset(csv_file_path, steps, input_labels) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
            dataset_input = dataset_input[0:20]
            block_sizes = [(6, 45), (6, 45), (4, 30), (4, 20)]
            dataset_input_loader = DataLoader(dataset_input)
        else:
            dataset_input = MyDataset(csv_file_path, steps, input_labels) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
            dataset_input = dataset_input[0:20]
            block_sizes = [(6, 45), (6, 45), (4, 30), (4, 20), (4, 20)]
            dataset_input_loader = DataLoader(dataset_input)
        x = next(iter(dataset_input_loader))
        output_size = (sum(block[0] for block in block_sizes), sum(block[1] for block in block_sizes))
        parser_encoder = argparse.ArgumentParser()
        parser_encoder.add_argument("--config", type=str, default="encoder_train_parameters.yaml")
        parser_encoder.add_argument("--debug", action="store_true")
        args_encoder = parser_encoder.parse_args()

        with open(args_encoder.config, "r") as f:
            config_encoder = yaml.load(f, Loader=yaml.FullLoader)
        config_encoder["model"]["p1"]["out_features"] = output_size[0]
        config_encoder["model"]["e1"]["synapse"]["out_features"] = output_size[1]

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

        pretrained_model = get_model(ModelMasked, config_encoder["model"], data=x[0, :], device=device, mask=mask_matrix)
        if separate_disturbances:
            pretrained_model.load_state_dict(torch.load("C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_control_input_parameters_masked_no_disturbances.pth"))
        else:
            pretrained_model.load_state_dict(torch.load("C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_control_input_parameters_masked.pth"))
        pretrained_dict = pretrained_model.state_dict()

        del(config_encoder["model"]["p1"])
        model = get_model(ModelEncoder, config_encoder["model"], data=x[0, :], device=device, mask=mask_matrix)
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

        if separate_disturbances:
            labels_disturbances = ["Mx", "My", "Mz", "Fz"]
            dataset_disturbances = MyDataset(csv_file_path, steps, labels_disturbances) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
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
            spikes_array_disturbances = spikes_array_disturbances.reshape(len(dataset_disturbances), steps, no_spiking_neurons_disturbances)
            dataset_input = torch.cat((dataset_input, torch.from_numpy(spikes_array_disturbances)), dim=2)
            dataset_input = dataset_input.float()

    print("Dataset_size input:", len(dataset_input), "Dataset_size output:", len(dataset_output))
    # Split the dataset into training and validation sets
    train_size = 0.9  # Adjust as needed
    train_dataset_input, val_dataset_input, train_dataset_output, val_dataset_output = train_test_split(dataset_input, dataset_output, train_size=train_size, shuffle=True, random_state=42)

    train_dataset = DatasetController(train_dataset_input, train_dataset_output)
    val_dataset = DatasetController(val_dataset_input, val_dataset_output)

    # Create a DataLoader
    train_loader = DataLoader(train_dataset, shuffle=True, **config["dataloader"])
    test_loader = DataLoader(val_dataset, shuffle=True, **config["dataloader"])

    # # get model and trace it
    x, y = next(iter(train_loader))
    print("Input size:", x[0, :].size(), "Output size:", y[0, :].size())
    model_controller = get_model(ModelController, config["model"], data=x[0, :], device=device)

    # logging with tensorboard
    if not args.debug:
        summary_writer = SummaryWriter(log_dir=("runs\\" + str(datetime.date.today()) + "_controller_e2e"))

    # optimizer
    optimizer = torch.optim.Adam(model_controller.parameters(), lr=config["training"]["lr"])
    optimizer.zero_grad()

    # loop that you can break
    try:
        with trange(config["training"]["epochs"], desc="epochs", leave=False, dynamic_ncols=True) as epochs_loop:
            for t in epochs_loop:
                # objgraph.show_growth(limit=10)
                train_loader = DataLoader(train_dataset, **config["dataloader"])#, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=400))
                train_loss = train_epoch(model_controller, train_loader, optimizer, device)
                test_loss = eval_model(model_controller, test_loader, device)

                epochs_loop.set_postfix_str(
                    f"train loss: {train_loss:.4f}, test loss: {test_loss:.4f},"
                    #f" test acc: {test_accuracy * 100:.2f}%"
                )
                if not args.debug:
                    summary_writer.add_scalar("train_loss", train_loss, t)
                    summary_writer.add_scalar("test_loss", test_loss, t)

    except KeyboardInterrupt:
        pass

    if not args.debug:
        summary_writer.flush()
        summary_writer.close()

    torch.save(model_controller.state_dict(), "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_controller_e2e_part2.pth")

    return model_controller


def model_pipeline(config_wandb, config_main, args):
    # tell wandb to get started
    with wandb.init(project="my_awesome_model", config=config_wandb):
        # make the model, data, and optimization problem
        model = main(config_main, args)
        print(model)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="controller_e2e_train.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_controller = main(config, args, spiking_input=True, separate_disturbances=False)
    # config_wandb = dict(
    #     epochs=config["training"]["epochs"],
    #     learning_rate=config["training"]["lr"],
    #     batch_size=config["dataloader"]["batch_size"],
    #     num_workers=config["dataloader"]["num_workers"]
    # )

    # model_pipeline(config_wandb, config, args)
