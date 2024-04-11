import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch

class MyDataset(Dataset):
    def __init__(self, csv_file, param, steps):
        self.data = pd.read_csv(csv_file, header=0, usecols=param)
        self.constant_columns = self.data.loc[:, :].columns[self.data.loc[:, :].std(axis=0) <= 1e-10]
        self.max_array = self.data.loc[:, :].max(axis=0)
        self.min_array = self.data.loc[:, :].min(axis=0)
        self.normalized_arr = self.data.copy()
        for col in self.data.columns:
            if col not in self.constant_columns:
                self.normalized_arr[col] = (self.data[col] - self.min_array[col]) / (self.max_array[col] - self.min_array[col])
        self.normalized_arr_reduced = self.normalized_arr.iloc[:, :]
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

def main(csv_file, testing_csv_file, labels, steps):
    # Create a dataset instance
    dataset = MyDataset(csv_file, labels, steps) #, transform=Compose([ToTensor(), ToBinTransform(), ToSeqTransform(steps)]))
    print("Dataset_size:", len(dataset))
    
    # Create a DataLoader
    # testing_dataset = MyDataset(testing_csv_file, labels, steps)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 20))
    fig.tight_layout(pad=4.0)
    for i, ax in enumerate(axes.flat):
        # for j in range(len(testing_dataset)):
        #     sequence = testing_dataset[j]
        #     ax.plot(sequence[:, i], color='red')
        for j in range(2048):
            sequence = dataset[j]
            ax.plot(sequence[:, i], color='blue')
        # ax.set_xlabel('Timestep')
        ax.grid(True)
        ax.set_title(labels[i])
    plt.show()


if __name__ == "__main__":
    csv_file_path = 'C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_control.csv'
    testing_csv_file_path = 'C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_test.csv'
    # labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4", 
    #           "Mx", "My", "Mz", "Fz", "gate_x", "gate_y", "gate_z", "gate_yaw", "u1", "u2", "u3", "u4"]
    
    labels = ["p", "phi", "vz"]

    main(csv_file_path, testing_csv_file_path, labels, 2000)
    