import numpy as np
import matplotlib.pyplot as plt

def leaky_integrate_and_fire(tau, dt, num_steps, I_ext, V_leak, I_leak):
    V = V_leak  # Initial membrane potential
    spike_times = []  # Record spike times

    for step in range(num_steps):
        dV = (I_ext - I_leak * (V - V_leak)) * dt / tau  # LIF equation
        V += dV
        print(V)

        if V >= 1.0:  # Spike condition
            spike_times.append(step * dt)
            V = V_leak  # Reset membrane potential after spike

    return spike_times

# Simulation parameters
tau = 20.0  # Membrane time constant (ms)
dt = 1.0  # Time step (ms)
num_steps = 1000  # Number of simulation steps
I_ext = 0.5  # External input current (nA)
V_leak = 0  # Leak voltage (mV)
I_leak = 0.1  # Leak current (nA)

# Run simulation
spike_times = leaky_integrate_and_fire(tau, dt, num_steps, I_ext, V_leak, I_leak)
print(spike_times)

# Plot results
plt.plot(spike_times, np.ones_like(spike_times), 'ro', markersize=5)
plt.title('Leaky Integrate-and-Fire Neuron')
plt.xlabel('Time (ms)')
plt.yticks([])
plt.show()
