import os
import numpy as np
import torch
import random
from torchvision import transforms
from normalize import Normalize, MapToRange
from torch.utils.data import Dataset, DataLoader

from torch import nn
# from torch_nn import *

from tqdm import tqdm
import time
import copy

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)

print(torch.cuda.is_available())

class TrajectoryDataset(Dataset):
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i, j = self.indices[index]        
        X = torch.tensor([
            self.dataset['dx'][i, j],
            self.dataset['dy'][i, j],
            self.dataset['dz'][i, j],
            self.dataset['vx'][i, j],
            self.dataset['vy'][i, j],
            self.dataset['vz'][i, j],
            self.dataset['phi'][i, j],
            self.dataset['theta'][i, j],
            self.dataset['psi'][i, j],
            self.dataset['p'][i, j],
            self.dataset['q'][i, j],
            self.dataset['r'][i, j],
            self.dataset['omega'][i, j, 0],
            self.dataset['omega'][i, j, 1],
            self.dataset['omega'][i, j, 2],
            self.dataset['omega'][i, j, 3],
            self.dataset['Mx_ext'][i],
            self.dataset['My_ext'][i],
            self.dataset['Mz_ext'][i]
        ], dtype=torch.float32)
        
        U = torch.tensor([
            self.dataset['u'][i, j, 0],
            self.dataset['u'][i, j, 1],
            self.dataset['u'][i, j, 2],
            self.dataset['u'][i, j, 3]
        ], dtype=torch.float32)
        
        return X, U
    
def main():
    # trajectories containing 199 points
    dataset_path = 'C:\\Users\\tavar\\Desktop\\Thesis\\Code\\quad_SL\\datasets\\hover_dataset.npz'

    dataset = dict()
    print('loading dataset...')
    with np.load(dataset_path) as full_dataset:
        # total number of trajectories
        num = len(full_dataset['dx'])
        print(num, 'trajectories')
        dataset = {key: full_dataset[key] for key in [
            't', 'dx', 'dy', 'dz', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r','omega', 'u', 'omega_min','omega_max', 'k_omega', 'Mx_ext', 'My_ext', 'Mz_ext'
        ]}

    # train/test split
    batchsize_train = 256
    batchsize_val = 4096
    train_trajectories = range(int(0.8*num))
    test_trajectories = list(set(range(int(0.99*num))) - set(train_trajectories))

    train_indices = [(i, j) for i in train_trajectories for j in range(199)]
    train_set = TrajectoryDataset(dataset, train_indices)
    train_loader = DataLoader(train_set, batch_size=batchsize_train, shuffle=True, num_workers=0)

    test_indices = [(i, j) for i in test_trajectories for j in range(199)]
    test_set = TrajectoryDataset(dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=batchsize_val, shuffle=True, num_workers=0)

    print('ready')

    print('Amount of testing trajectories: ',len(test_trajectories),f'(Batchsize: {batchsize_val})')
    print('Amount of training trajectories: ',len(train_trajectories),f'(Batchsize: {batchsize_train})')

    print(dataset['omega_min'])
    print(dataset['omega_max'])
    X_mean = torch.zeros(19)
    X_std = torch.zeros(19)

    N=100000

    for i, data in tqdm(enumerate(test_set)):
        X = data[0]
        X_mean += X
        if i>=N:
            break
    X_mean = X_mean/N

    print('mean:')
    print(X_mean)
        
    for i, data in tqdm(enumerate(test_set)):
        X = data[0]
        X_std += (X-X_mean)**2
        if i>=N:
            break

    X_std = torch.sqrt(X_std/N)
    print('std:')
    print(X_std)

    model = nn.Sequential(
        Normalize(mean=X_mean, std=X_std),
        nn.Linear(19, 120),
        nn.ReLU(),
        nn.Linear(120, 120),
        nn.ReLU(),
        nn.Linear(120, 120),
        nn.ReLU(),
        nn.Linear(120, 4),
        nn.Sigmoid()
    )


    x1 = torch.randn(19)
    print(model(x1))
    # print([param.shape for param in model.parameters()])

    criterion = torch.nn.MSELoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1, verbose=True, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    loss_list = []
    loss_val_list = []
    best_loss = 0.1
    first = True
    start_time = time.time()

    # loop over the dataset multiple times
    num_epochs = 30

    # nn_model_name = f"{dataset_path[9:-4]}_{batchsize_train}_{batchsize_val}_{learning_rate}_{num_epochs}"

    for epoch in range(num_epochs):
        
        if first:
            time_remaining = '-'
        else:
            time_estimate = epoch_time*(num_epochs-epoch+1)
            if time_estimate > 60:
                if time_estimate > 3600:
                    time_remaining = str(round(time_estimate/3600,2))+' h'
                else:
                    time_remaining = str(round(time_estimate/60,2))+' min'
            else:
                time_remaining = str(round(time_estimate,0))+' s'
            
        first = False
        print(f"Epoch {epoch+1}/{num_epochs}, Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}, Time remaining: {time_remaining}")

        start_time_epoch = time.time()
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        
        for i, (data, targets) in loop:
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            
            # Loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update progressbar
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
            loss_list.append(loss.item())

        # Validate
        with torch.no_grad():
            # Get a random batch from the test dataset
            data_val, targets_val = next(iter(test_loader))

            # Forward pass
            outputs_val = model(data_val)

            # Loss
            loss_val = criterion(outputs_val, targets_val)

            if loss_val < best_loss:
                # Save best model
                best_model = copy.deepcopy(model)
                
                # Backup
                torch.save(model.state_dict(), './quad_SL/neural_networks/model_whole_dataset.pth')
                
                best_loss = loss_val
                print("Best model updated!")

            # Scheduler (reduce learning rate if loss stagnates)
            scheduler.step(loss_val)
            
            loss_val_list.append(loss_val.item())

        print(f'loss = {loss:.8f}, loss validation = {loss_val:.8f} '+r' (control error: +/-'+str(round(100*np.sqrt(float(loss_val)),2))+'%)\n')

        epoch_time = (time.time() - start_time_epoch)

        loop.close()
        
    # Compute excecution time
    execution_time = (time.time() - start_time)    
    print(f"Total training time: {round(execution_time,2)}s")

    # Save best model and copy for maptorange network
    # torch.save(best_model.state_dict(), f'neural_networks/{nn_model_name}.pth')
    best_model_for_maptorange = model
    best_model_for_maptorange.load_state_dict(torch.load('./quad_SL/neural_networks/model_whole_dataset.pth'))
    print(best_model_for_maptorange)

if __name__ == '__main__':
    main()