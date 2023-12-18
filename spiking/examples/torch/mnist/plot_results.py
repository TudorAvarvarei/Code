from tensorboard.backend.event_processing import event_accumulator
from matplotlib import pyplot as plt
import os

# Specify the path to your TensorBoard log directory
log_dirs = ['./runs/2023-12-04_neurons_200_epochs_10',
           './runs/2023-12-04_neurons_300_epochs_10',
           './runs/2023-12-04_neurons_400_epochs_10',
           './runs/2023-12-01_neurons_500_epochs_10',
           './runs/2023-12-04_neurons_600_epochs_10',
           './runs/2023-12-04_neurons_700_epochs_10',
           './runs/2023-12-04_neurons_800_epochs_10']

epochs = [200, 300, 400, 500, 600, 700, 800]

train_loss_dict = {}
test_loss_dict = {}
i=0

for log_dir in log_dirs:
    train_loss_dict[epochs[i]]=[]
    test_loss_dict[epochs[i]]=[]
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
            train_loss_dict[epochs[i]].append(event.value)
        for event in test_loss_data:
            test_loss_dict[epochs[i]].append(event.value)

    i += 1

fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
for epoch in epochs:
    ax1.plot(train_loss_dict[epoch], label=str(epoch))

for epoch in epochs:
    ax2.plot(test_loss_dict[epoch], label=str(epoch))

fig.text(0.5, 0.04, 'Epoch', ha='center')
fig.text(0.04, 0.5, 'Loss', va='center', rotation='vertical')
ax1.set_title("Training data")
ax1.legend()
ax2.set_title("Testing data")
ax2.legend()

fig.show()
plt.pause(9999)
