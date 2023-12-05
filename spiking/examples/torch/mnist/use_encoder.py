import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import sys
sys.path.append("C:\\Users\\tavar\\Desktop\\Thesis\\Code\\spiking_tutorial\\spiking\\core\\torch")
sys.path.append("C:\\Users\\tavar\\Desktop\\Thesis\\Code\\spiking_tutorial\\spiking")
from examples.torch.mnist.encoder_train import Model
from core.torch.model import get_model

# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=0)
        self.max_array = np.max(self.data, axis=0)
        self.min_array = np.min(self.data, axis=0)
        self.normalized_arr = (self.data - self.min_array) / (self.max_array - self.min_array)
        self.normalized_arr_reduced = self.normalized_arr.iloc[0:5, :]

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
    csv_file_path = 'data_test.csv'

    # Create a dataset instance
    dataset = MyDataset(csv_file_path) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    print("Dataset_size:", len(dataset))
    
    # Create a DataLoader
    testing_dataset = DataLoader(dataset, shuffle=False)

    # # get model and trace it
    x = next(iter(testing_dataset))
    print("Input size:", x[0, :].size())
    model = get_model(Model, config["model"], data=x[0, :], device=device)
    model.load_state_dict(torch.load("C:/Users/tavar/Desktop/Thesis/Code/spiking_tutorial/spiking/examples/torch/mnist/models/model.pth"))
    # model = torch.load("C:/Users/tavar/Desktop/Thesis/Code/spiking_tutorial/spiking/examples/torch/mnist/models/model2.pt")

    model.eval()
    with torch.no_grad():
        with tqdm(testing_dataset) as batches_loop:
            for data in batches_loop:
                data = data.to(device)
                data = data.swapaxes(0, 1)  # (b, t, c, h, w) -> (t, b, c, h, w)
                for x, y in zip(data, data):
                    yhat = model(x)
                    print("y_real", y)
                    print("y_predicted", yhat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="encoder_train.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)