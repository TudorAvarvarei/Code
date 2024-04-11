'''
import numpy as np

def encode_spikes(inputs, time_period=100, max_rate=100):
    """
    Encode float inputs into spikes using rate-based encoding.

    Parameters:
    - inputs: List of 26 float values.
    - time_period: Time period in milliseconds for encoding.
    - max_rate: Maximum firing rate of neurons.

    Returns:
    - List of spike trains for each input.
    """
    spike_trains = []

    for value in inputs:
        firing_rate = max(0, min(value, max_rate))  # Ensure firing rate is within [0, max_rate]
        spike_train = np.random.rand(time_period) < (firing_rate / max_rate)
        spike_trains.append(spike_train.astype(int))

    return spike_trains

# Example usage:
float_inputs = [0.5, 0.8, 1.2, 0.1, 0.7, 0.3, 1.0, 0.9, 0.4, 0.6, 0.2, 0.8, 0.7, 0.5, 0.3, 0.1, 1.0, 0.6, 0.4, 0.9, 0.2, 0.7, 0.3, 0.8, 1.0, 0.5]
spike_output = encode_spikes(float_inputs)

# Print the spike trains
for i, spike_train in enumerate(spike_output):
    print(f"Input {i + 1}: {spike_train}")
'''

"""import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense

# Generate random data for training
num_samples = 10000
input_size = 26
data = np.random.rand(num_samples, input_size)

# Define the autoencoder model
model = Sequential()
model.add(Dense(64, input_dim=input_size, activation='relu'))
model.add(Dense(32, activation='relu'))  # Adjust the size of the encoded representation as needed
model.add(Dense(64, activation='relu'))
model.add(Dense(input_size, activation='sigmoid'))  # Sigmoid activation for values between 0 and 1

model.compile(optimizer='adam', loss='mse')  # Mean Squared Error loss for reconstruction

# Train the autoencoder
model.fit(data, data, epochs=50, batch_size=32)

# Create a new model with only the encoder part
# encoder_model = Model(inputs=model.input, outputs=model.layers[2].output)

# Use the trained autoencoder for encoding and decoding
encoded_data = model.predict(data)

# Print some results
print("Original Data:")
print(data[0])
print("\nEncoded Data:")
print(encoded_data[0])"""

# import pandas as pd

# df = pd.read_csv('C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_refined.csv', header=0)

# df = df.drop(range(2252000, 2254000, 1))
# df = df.drop(range(3616000, 3618000, 1))

# df.to_csv('C:/Users/tavar/Desktop/Thesis/Code/spiking/examples/torch/mnist/data/data_refined.csv', index=False)

"""
import os
from tensorboard.backend.event_processing import event_accumulator

log_dir='./runs/2024-02-29large_leaks_trained_long'
# Find the latest event file in the log directory
event_file = max([os.path.join(log_dir, d) for d in os.listdir(log_dir) if d.startswith("events")])

# Create an EventAccumulator
event_acc = event_accumulator.EventAccumulator(event_file)

# Load the TensorBoard events
event_acc.Reload()

# Get the available keys
available_keys = event_acc.Tags()

# Access data associated with a specific key
if 'train_loss' in available_keys['scalars']:
    train_loss_data = event_acc.Scalars('train_loss')
    test_loss_data = event_acc.Scalars('test_loss')
    for event in train_loss_data:
        print(event.value)
    print("\n")
    for event in test_loss_data:
        print(event.value)
"""

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# from scipy.interpolate import CubicSpline

# # Gate positions
# # gate_positions = [
# #     [2, -2.5, 1],
# #     [3, 0, 1.5],
# #     [0.5, 0, 1.5],
# #     [-2, -1.5, 2],
# #     [-2.5, 1, 2],
# #     [0, 1.5, 1.5],
# #     [2, 2, 2],
# #     [3, 0, 1.5],
# #     [-1, 0, 1.5],
# #     [-3.5, 0, 1.5],
# #     [-1.5, -3.5, 1.5],
# #     [1, -1.5, 1],
# #     [3.5, -3, 1.5],
# #     [2, -2.5, 1]
# # ]

# gate_positions = [
#     [ 2,-1.5,-1.5],
#     [ 2, 1.5,-1.5],
#     [-2, 1.5,-1.5],
#     [-2,-1.5,-1.5],
#     [ 2,-1.5,-1.5]
# ]

# # Gate orientations
# # gate_orientations = [
# #     np.pi / 2,
# #     np.pi / 2,
# #     3 * np.pi / 2,
# #     3 * np.pi / 4, 
# #     np.pi / 4,
# #     0,
# #     0,
# #     3 * np.pi / 2,
# #     np.pi,
# #     3 * np.pi / 2,
# #     0,
# #     np.pi / 2,
# #     3 * np.pi / 2,
# #     np.pi / 2
# # ]

# gate_orientations = [
#     np.pi/4,
#     3*np.pi/4,
#     5*np.pi/4,
#     7*np.pi/4,
#     np.pi/4
# ]

# # Extract x, y, z coordinates from gate positions
# x, y, z = zip(*gate_positions)

# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the gates
# # ax.scatter(x, y, z, c='r', marker='o')

# # Create a cubic spline interpolation for x, y, and z coordinates
# cs_x = CubicSpline(np.arange(len(x)), x)
# cs_y = CubicSpline(np.arange(len(y)), y)
# cs_z = CubicSpline(np.arange(len(z)), z)

# # Plot fully continuous line connecting the gates
# num_points = 100  # Adjust as needed for smoothness
# s = np.linspace(0, len(x) - 1, num_points * (len(x) - 1))
# ax.plot(cs_x(s), cs_y(s), cs_z(s), color='b')

# # Plot gate orientations as arrows
# # for i, (pos, orient) in enumerate(zip(gate_positions, gate_orientations)):
# #     ax.quiver(pos[0], pos[1], pos[2], np.cos(orient), np.sin(orient), 0, length=0.5, color='g')

# # Define the size of the rectangle
# rectangle_width = 0.5
# rectangle_height = 0.5
# for i, (position, orientation) in enumerate(zip(gate_positions, gate_orientations)):
#     # Define the vertices of the rectangle
#     vertices = np.array([
#         [position[0] + np.cos(orientation - np.pi / 2) * rectangle_width / 2, 
#          position[1] + np.sin(orientation - np.pi / 2) * rectangle_width / 2, 
#          position[2] - rectangle_height / 2],
#         [position[0] + np.cos(orientation + np.pi / 2) * rectangle_width / 2, 
#          position[1] + np.sin(orientation + np.pi / 2) * rectangle_width / 2, 
#          position[2] - rectangle_height / 2],
#         [position[0] + np.cos(orientation + np.pi / 2) * rectangle_width / 2, 
#          position[1] + np.sin(orientation + np.pi / 2) * rectangle_width / 2, 
#          position[2] + rectangle_height / 2],
#         [position[0] + np.cos(orientation - np.pi / 2) * rectangle_width / 2, 
#          position[1] + np.sin(orientation - np.pi / 2) * rectangle_width / 2, 
#          position[2] + rectangle_height / 2],
#         [position[0] + np.cos(orientation - np.pi / 2) * rectangle_width / 2, 
#          position[1] + np.sin(orientation - np.pi / 2) * rectangle_width / 2, 
#          position[2] - rectangle_height / 2]
#     ])
    
#     # Plot rectangle
#     ax.plot(vertices[:,0], vertices[:,1], vertices[:,2], color='orange')

#     # Annotate with gate number
#     if i not in [4]:
#         ax.text(position[0], position[1], position[2] + rectangle_height, str(i + 1), color='red', fontsize=12)
#     elif (i == 1):
#         ax.text(position[0], position[1], position[2] + rectangle_height, "2/8", color='red', fontsize=12)

# # Labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Testing Track')
# ax.set_aspect('equal')
# ax.view_init(elev=90, azim=0)

# plt.show()

import matplotlib.pyplot as plt

labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4", 
          "Mx", "My", "Mz", "Fz", "gate_x", "gate_y", "gate_z", "gate_yaw", "u1", "u2", "u3", "u4"]

at_once = {'x': 0.055736408, 'y': 0.044692323, 'z': 0.053857062, 'vx': 0.05906153, 'vy': 0.065190874, 'vz': 0.0408006, 
           'phi': 0.04794844, 'theta': 0.041651208, 'psi': 0.075179264, 'p': 0.048515547, 'q': 0.035467044, 'r': 0.053818513, 
           'w1': 0.054159813, 'w2': 0.044427726, 'w3': 0.05422948, 'w4': 0.060256585, 'Mx': 0.07205693, 'My': 0.11504696, 'Mz': 0.07644945, 
           'Fz': 0.069524504, 'gate_x': 0.0385971, 'gate_y': 0.02322253, 'gate_z': 0.036343813, 'gate_yaw': 0.021013513, 
           'u1': 0.062029053, 'u2': 0.05029848, 'u3': 0.06452216, 'u4': 0.06083837}

batches = {'x': 0.02790151, 'y': 0.025045095, 'z': 0.023679191, 'vx': 0.029735671, 'vy': 0.027135199, 'vz': 0.024110133,
           'phi': 0.024214504, 'theta': 0.023538098, 'psi': 0.027750256, 'p': 0.027427524, 'q': 0.024677847, 'r': 0.024158109,
           'w1': 0.043472435, 'w2': 0.031815764, 'w3': 0.035546567, 'w4': 0.03258463,
           'Mx': 0.010618235603455158, 'My': 0.011128867537744527, 'Mz': 0.009189414978026882, 'Fz': 0.008842261952739221,
           'gate_x': 0.00018011351, 'gate_y': 0.00014677424, 'gate_z': 6.4841784e-05, 'gate_yaw': 0.0001055058, 
           'u1': 0.0835511, 'u2': 0.06660617, 'u3': 0.06329836, 'u4': 0.07227252}

values = list(batches.values())
values = values[:-4]
print(sum(values) / len(values))

# fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(20, 20))
# fig.tight_layout(pad=4.0)
# fig.suptitle("Comparison training at once vs batches", fontsize=16)
# bar_labels = ("All_at_once", "Batches")
# plt.rcParams.update({'font.size': 12})
# for i, ax in enumerate(axes.flat):
#     ax.bar("All_at_once", list(at_once.values())[i], width=0.5)
#     ax.bar("Batches", list(batches.values())[i], width=0.5)
#     # ax.set_xlabel('Timestep')
#     ax.grid(True)
#     ax.set_title(labels[i])
#     # ax.legend(loc='best', fontsize=12)
# plt.show()
