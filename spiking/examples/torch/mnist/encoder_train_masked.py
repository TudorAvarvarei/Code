import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
from scipy.sparse import lil_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import yaml

sys.path.append("C:\\Users\\tavar\\Desktop\\Thesis\\Code\\spiking\\core\\torch")
sys.path.append("C:\\Users\\tavar\\Desktop\\Thesis\\Code\\spiking")
from core.torch.layer import MaskedLinearCubaLif
from core.torch.model import get_model, BaseModel
from core.torch.synapse import MaskedLazyLinear
import wandb

# spikes = []

class ModelMasked(BaseModel):
    def __init__(self, e1, p1, mask=None):
        super().__init__()

        self.e1 = MaskedLinearCubaLif(**e1, mask=torch.t(mask))
        self.p1 = MaskedLazyLinear(**p1, mask=mask)

    def forward(self, input):
        x = self.e1(input)
        # spikes.append(x.tolist())
        x = x.flatten(start_dim=1)
        x = self.p1(x)
        return x[0]

    def reset(self):
        self.e1.reset()

def sequence(model, data):
    data = data.swapaxes(0, 1)  # (b, t, c, h, w) -> (t, b, c, h, w)

    model.reset()
    loss = 0
    
    for x, y in zip(data, data):
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
        for data in batches_loop:
            data = data.to(device)
            optimizer.step()
            optimizer.zero_grad()

            loss = sequence(model, data)
            loss.backward()

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
            for data in batches_loop:
                data = data.to(device)

                loss = sequence(model, data)

                epoch_loss += loss.item()
                # epoch_accuracy += accuracy.item()
                passes += 1

                batches_loop.set_postfix_str(
                    f"test loss: {epoch_loss / passes:.4f}" #, test accuracy: {epoch_accuracy / passes * 100:.2f}"
                )

    return epoch_loss / len(dataloader) #, epoch_accuracy / len(dataloader)
    
# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, csv_file, steps, parameters):
        self.data = pd.read_csv(csv_file, header=0, usecols=parameters)
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

def main(config, args):
    # training params
    device = "cpu"
    device = torch.device(device)
    steps = config["dataset"]["steps"]

    # Specify the path to your CSV file
    csv_file_path = 'C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_control.csv'

    # labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4", 
    #           "Mx", "My", "Mz", "Fz", "gate_x", "gate_y", "gate_z", "gate_yaw", "u1", "u2", "u3", "u4"]

    parameters = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4", 
                  "gate_x", "gate_y", "gate_z", "gate_yaw"]

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
    
    # Create a dataset instance
    dataset = MyDataset(csv_file_path, steps, parameters) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    # dataset = dataset[0:1024]
    print("Dataset_size:", len(dataset))
    # Split the dataset into training and validation sets
    train_size = 0.9  # Adjust as needed
    train_dataset, val_dataset = train_test_split(dataset, train_size=train_size, shuffle=True, random_state=42)

    # Create a DataLoader
    train_loader = DataLoader(train_dataset, shuffle=True, **config["dataloader"])
    test_loader = DataLoader(val_dataset, shuffle=True, **config["dataloader"])

    # # get model and trace it
    x = next(iter(train_loader))
    print("Input size:", x[0, :].size())
    model = get_model(ModelMasked, config["model"], data=x[0, :], device=device, mask=mask_matrix)

    # logging with tensorboard
    if not args.debug:
        summary_writer = SummaryWriter(log_dir=("runs\\" + str(datetime.date.today()) + "encode_input_param_no_disturb"))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    optimizer.zero_grad()

    # loop that you can break
    # best_accuracy = 1000000000
    try:
        with trange(config["training"]["epochs"], desc="epochs", leave=False, dynamic_ncols=True) as epochs_loop:
            for t in epochs_loop:
                # objgraph.show_growth(limit=10)
                train_loader = DataLoader(train_dataset, **config["dataloader"])#, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=400))
                train_loss = train_epoch(model, train_loader, optimizer, device)
                test_loss = eval_model(model, test_loader, device)

                epochs_loop.set_postfix_str(
                    f"train loss: {train_loss:.4f}, test loss: {test_loss:.4f},"
                    #f" test acc: {test_accuracy * 100:.2f}%"
                )
                if not args.debug:
                    summary_writer.add_scalar("train_loss", train_loss, t)
                    summary_writer.add_scalar("test_loss", test_loss, t)
                
                # if test_loss < best_accuracy:
                #     best_accuracy = test_loss
                #     print("Epoch:", t, "Test_loss", test_loss)
                #     torch.save(model.state_dict(), "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_control_input_parameters_masked.pth")

    except KeyboardInterrupt:
        pass

    if not args.debug:
        summary_writer.flush()
        summary_writer.close()

    torch.save(model.state_dict(), "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_control_input_parameters_masked_no_disturbances.pth")

    return model


def model_pipeline(config_wandb, config_main, args):
    # tell wandb to get started
    with wandb.init(project="my_awesome_model", config=config_wandb):
        # make the model, data, and optimization problem
        model = main(config_main, args)
        print(model)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="encoder_train_parameters.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config, args)
    # config_wandb = dict(
    #     epochs=config["training"]["epochs"],
    #     learning_rate=config["training"]["lr"],
    #     batch_size=config["dataloader"]["batch_size"],
    #     num_workers=config["dataloader"]["num_workers"]
    # )

    # model_pipeline(config_wandb, config, args)
