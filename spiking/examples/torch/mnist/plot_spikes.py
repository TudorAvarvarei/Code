from matplotlib import pyplot as plt
import torch
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
from core.torch.layer import LinearCubaLif
from examples.torch.mnist.encoder_train_masked import ModelMasked
from core.torch.layer import MaskedLinearCubaLif


class ModelEncoder(BaseModel):
    def __init__(self, e1, mask=None):
        super().__init__()

        self.e1 = MaskedLinearCubaLif(**e1, mask=torch.t(mask))

    def forward(self, input):
        x = self.e1(input)
        x = x.flatten(start_dim=1)
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

def main(config):
    # Get device
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     print("Device = CUDA")
    # else:
    #     device = "cpu"
    #     print("Device = CPU")
    device = torch.device("cpu")

    # Specify the path to your CSV file
    csv_file_path = 'C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_control_test.csv'
    train_csv_file_path = 'C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_control.csv'

    # labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4", 
    #           "Mx", "My", "Mz", "Fz", "gate_x", "gate_y", "gate_z", "gate_yaw", "u1", "u2", "u3", "u4"]
    
    labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4", 
              "gate_x", "gate_y", "gate_z", "gate_yaw"]
    
    steps = config["dataset"]["steps"]

    # Create a dataset instance
    dataset = MyDataset(train_csv_file_path, csv_file_path, labels) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    print("Dataset_size:", len(dataset))
    
    # Create a DataLoader
    testing_dataset = DataLoader(dataset, shuffle=False)

    # # get model and trace it
    x = next(iter(testing_dataset))
    print("Input size:", x[0, :].size())

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
    pretrained_model.load_state_dict(torch.load("C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_control_input_parameters_masked_no_disturbances.pth"))
    pretrained_dict = pretrained_model.state_dict()

    del(config["model"]["p1"])
    model = get_model(ModelEncoder, config["model"], data=x[0, :], device=device, mask=mask_matrix)
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)

    model.eval()
    save_list = []

    for name, param in pretrained_model.named_parameters():
        print(name, ": ", param)


    with torch.no_grad():
        with tqdm(testing_dataset) as batches_loop:
            for data in batches_loop:
                data = data.to(device)
                data = data.swapaxes(0, 1)  # (b, t, c, h, w) -> (t, b, c, h, w)
                for x, y in zip(data, data):
                    yhat = model(x)
                    save_list.append(yhat)
    
    tensor_list = torch.cat(save_list, dim=0)
    tensor_list_gates = tensor_list[:, 45:90]
    print(tensor_list_gates.shape)
    fig, axes = plt.subplots(nrows=9, ncols=5, figsize=(20, 20))
    fig.tight_layout(pad=4.0)
    fig.suptitle("Spiking activity for Angles + Angular velocities")
    for i, ax in enumerate(axes.flat):
        ax.plot(tensor_list_gates[:, i].cpu().numpy(), label='Real')
        ax.set_title("Neuron " + str(i + 1))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="encoder_train_parameters.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)