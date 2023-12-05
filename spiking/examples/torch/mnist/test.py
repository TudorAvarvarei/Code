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

import numpy as np
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
print(encoded_data[0])
