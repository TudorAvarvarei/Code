import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

def get_dataset(csv_file, labels):
    data = pd.read_csv(csv_file, header=0, usecols=labels)
    constant_columns = data.loc[:, :].columns[data.loc[:, :].std(axis=0) <= 1e-10]
    max_array = data.loc[:, :].max(axis=0)
    min_array = data.loc[:, :].min(axis=0)
    normalized_arr = data.copy()
    for col in data.columns:
        if col not in constant_columns:
            normalized_arr[col] = (data[col] - min_array[col]) / (max_array[col] - min_array[col])
    values = normalized_arr.values
    length = int(len(data))
    normalized_final_array = values.reshape(length, len(values[0]))
    return normalized_final_array

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(24, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 120)
        self.fc4 = nn.Linear(120, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def main():
    # Specify the path to your CSV file
    csv_file_path = 'C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_control.csv'

    input_labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4",
                    "Mx", "My", "Mz", "Fz", "gate_x", "gate_y", "gate_z", "gate_yaw"]

    output_labels = ["u1", "u2", "u3", "u4"]

    inputs = get_dataset(csv_file_path, input_labels)
    outputs = get_dataset(csv_file_path, output_labels)

    # Convert numpy arrays to PyTorch tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(inputs_tensor, outputs_tensor, test_size=0.1, random_state=42)

    # Create DataLoader for training and testing sets
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = NeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        print("Start training epoch: ", epoch+1)
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Evaluate the model
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    torch.save(model.state_dict(), "C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/models/model_controller_chatGPT_250_neurons.pth")

if __name__ == "__main__":
    main()