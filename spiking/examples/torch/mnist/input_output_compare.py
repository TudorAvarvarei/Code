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
import matplotlib.pyplot as plt

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
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=0)
        self.constant_columns = self.data.loc[:, :].columns[self.data.loc[:, :].std(axis=0) <= 1e-10]
        self.max_array = self.data.loc[:, :].max(axis=0)
        self.min_array = self.data.loc[:, :].min(axis=0)
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

def main(config):
    # Get device
    if torch.cuda.is_available():
        device = "cuda:0"
        print("Device = CUDA")
    else:
        device = "cpu"
        print("Device = CPU")
    device = torch.device(device)

    # Specify the path to your CSV file
    csv_file_path = 'C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_test.csv'

    # Create a dataset instance
    dataset = MyDataset(csv_file_path) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    print("Dataset_size:", len(dataset))
    
    # Create a DataLoader
    testing_dataset = DataLoader(dataset, shuffle=False)

    # # get model and trace it
    x = next(iter(testing_dataset))
    print("Input size:", x[0, :].size())
    pretrained_model = get_model(Model, config["model"], data=x[0, :], device=device)
    pretrained_model.load_state_dict(torch.load("C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_200_neurons.pth"))
    pretrained_dict = pretrained_model.state_dict()

    model = get_model(Model, config["model"], data=x[0, :], device=device)
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

    labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4", 
              "Mx", "My", "Mz", "Fz", "gate_x", "gate_y", "gate_z", "gate_yaw", "u1", "u2", "u3", "u4"]
    
    average_error_dict = dict(zip(labels, average_error))
    print(average_error_dict)

    fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(20, 20))
    fig.tight_layout(pad=4.0)
    for i, ax in enumerate(axes.flat):
        ax.plot(input[:, i], label='Real')
        ax.plot(output[:, i], label='Estimated')
        # ax.set_xlabel('Timestep')
        ax.grid(True)
        ax.set_title(labels[i])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="encoder_train.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)