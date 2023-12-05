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
from core.torch.layer import LinearCubaLif
from core.torch.model import get_model, BaseModel

# spikes = []

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
    def __init__(self, csv_file, steps):
        self.data = pd.read_csv(csv_file, header=0)
        self.max_array = np.max(self.data, axis=0)
        self.min_array = np.min(self.data, axis=0)
        self.normalized_arr = (self.data - self.min_array) / (self.max_array - self.min_array)
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
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]
    # device = config["training"]["device"]
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     print("Device = CUDA")
    # else:
    #     device = "cpu"
    #     print("Device = CPU")

    device = "cpu"

    device = torch.device(device)

    # dataset
    # make single image into a sequence repeating the image for the number of steps
    steps = config["dataset"]["steps"]
    # x_to_bin = lambda x: x.gt(0.5).float()
    # x_to_seq = lambda x: x.unsqueeze(0).expand(steps, -1, -1, -1)  # (c, h, w) -> (t, c, h, w)
    # train_ds = MNIST("data", download=True, train=True, transform = Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    # test_ds = MNIST("data", download=True, train=False, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    
    # train_ds = MNIST("data", download=True, train=True, transform = Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    # test_ds = MNIST("data", download=True, train=False, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))

    # dataloader
    # train_loader = DataLoader(train_ds, shuffle=True, **config["dataloader"])
    # test_loader = DataLoader(test_ds, shuffle=False, **config["dataloader"])

    # Specify the path to your CSV file
    csv_file_path = 'C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data.csv'

    # Create a dataset instance
    dataset = MyDataset(csv_file_path, steps) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    print("Dataset_size:", len(dataset))
    # Split the dataset into training and validation sets
    train_size = 0.9  # Adjust as needed
    train_dataset, val_dataset = train_test_split(dataset, train_size=train_size, shuffle=True, random_state=42)
    # print("train:", len(train_dataset), "validate:", len(val_dataset))

    # Create a DataLoader
    train_loader = DataLoader(train_dataset, shuffle=True, **config["dataloader"])
    test_loader = DataLoader(val_dataset, shuffle=True, **config["dataloader"]) 

    # # get model and trace it
    x = next(iter(train_loader))
    print("Input size:", x[0, :].size())
    model = get_model(Model, config["model"], data=x[0, :], device=device)

    # logging with tensorboard
    if not args.debug:
        summary_writer = SummaryWriter(log_dir=("runs\\" + str(datetime.date.today()) + "_neurons_" + str(config["model"]["e1"]["synapse"]["out_features"]) 
                                                + "_epochs_" + str(config["training"]["epochs"])))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()

    # loop that you can break
    try:
        with trange(epochs, desc="epochs", leave=False, dynamic_ncols=True) as epochs_loop:
            for t in epochs_loop:
                objgraph.show_growth(limit=10)
                train_loss = train_epoch(model, train_loader, optimizer, device)
                test_loss = eval_model(model, test_loader, device)

                epochs_loop.set_postfix_str(
                    f"train loss: {train_loss:.4f}, test loss: {test_loss:.4f},"
                    #f" test acc: {test_accuracy * 100:.2f}%"
                )
                if not args.debug:
                    summary_writer.add_scalar("train_loss", train_loss, t)
                    #summary_writer.add_scalar("train_accuracy", train_accuracy, t)
                    summary_writer.add_scalar("test_loss", test_loss, t)
                    #summary_writer.add_scalar("test_accuracy", test_accuracy, t)

    except KeyboardInterrupt:
        pass

    if not args.debug:
        summary_writer.flush()
        summary_writer.close()

    torch.save(model.state_dict(), "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_CPU_800_neurons.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="encoder_train.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config, args)
