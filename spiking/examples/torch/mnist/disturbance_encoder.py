import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    
def encoder(no_spiking_neurons, values):
    values = values.cpu().numpy()
    values = np.squeeze(values)
    neuron_per_parameter = no_spiking_neurons // len(values)
    intervals = np.linspace(0, 1, num=(2**neuron_per_parameter))
    closest_indices = np.zeros(len(values), dtype=int)
    for i, value in enumerate(values):
        distances = np.abs(intervals - value)
        closest_indices[i] = np.argmin(distances)
    # encoded_output = closest_indices / ((2**neuron_per_parameter)- 1)
    binary_indices = np.array([np.binary_repr(x, width=neuron_per_parameter) for x in closest_indices])
    binary_arrays = np.array([[float(bit) for bit in binary_str] for binary_str in binary_indices])
    binary_arrays = binary_arrays.flatten()
    return torch.tensor(binary_arrays)

def decoder(spikes, labels):
    neurons_per_parameter = len(spikes) // len(labels)
    output_values = np.zeros(len(labels), dtype=float)
    common_divisor = 2**(neurons_per_parameter) - 1
    for i in range(len(labels)):
        value = 0
        for j in range(neurons_per_parameter):
            value += ((2**(neurons_per_parameter - j - 1) / common_divisor) * spikes[(i*neurons_per_parameter) + j])
        output_values[i] = value
    return torch.tensor(output_values)

def main(no_spiking_neurons, labels):
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
    # values = torch.Tensor([0.4, 0.8, 0.232, 0.245])
    # print(encoder(no_spiking_neurons, values))

    error = []
    input = []
    spikes_array = []
    output = []
    # Create a DataLoader
    testing_dataset = DataLoader(dataset, shuffle=False)
    with torch.no_grad():
        with tqdm(testing_dataset) as batches_loop:
            for data in batches_loop:
                data = data.to(device)
                data = data.swapaxes(0, 1)  # (b, t, c, h, w) -> (t, b, c, h, w)
                for x, y in zip(data, data):
                    spikes = encoder(no_spiking_neurons, x)
                    yhat = decoder(spikes, labels)
                    error.append(abs(y-yhat).cpu().numpy())
                    input.append(x.cpu().numpy())
                    spikes_array.append(spikes.cpu(). numpy())
                    output.append(yhat.cpu().numpy())
                    # print("y_real", y)
                    # print("y_predicted", yhat)
    error = np.squeeze(np.array(error))
    input = np.squeeze(np.array(input))
    output = np.squeeze(np.array(output))
    average_error = np.average(error, axis=0)
    
    average_error_dict = dict(zip(labels, average_error))
    print(average_error_dict)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    fig.tight_layout(pad=4.0)
    fig.suptitle("Input parameters")
    for i, ax in enumerate(axes.flat):
        ax.plot(input[:, i], label='Real')
        ax.plot(output[:, i], label='Estimated')
        # ax.set_xlabel('Timestep')
        ax.grid(True)
        ax.set_title(labels[i])
    plt.show()
    spikes_array = torch.tensor(spikes_array)

    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 20))
    fig.tight_layout(pad=4.0)
    fig.suptitle("Activity of angles")
    for i, ax in enumerate(axes.flat):
        ax.plot(spikes_array[:, i].cpu().numpy(), label='Real')
        ax.set_title("Neuron " + str(i + 1))
    plt.show()

    pass


if __name__ == "__main__":
    no_spiking_neurons = 20
    
    labels = ["Mx", "My", "Mz", "Fz"]

    main(no_spiking_neurons, labels)