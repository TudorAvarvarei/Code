from tensorboard.backend.event_processing import event_accumulator
from matplotlib import pyplot as plt
import os

# Specify the path to your TensorBoard log directory
log_dirs = ['./runs/2024-02-27_batches_64',
            './runs/2024-02-27_batches_128',
            './runs/2024-02-27_batches_256',
            './runs/2024-02-27_batches_512',
            './runs/2024-02-27_batches_1024',
            './runs/2024-02-27_batches_2048']


epochs = [0, 1, 2, 3, 4, 5]
legend = ["64_batches",
          "128_batches",
          "256_batches",
          "512_batches",
          "1024_batches",
          "2048_batches",]

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
    ax1.plot(train_loss_dict[epoch], label=legend[epoch])

for epoch in epochs:
    ax2.plot(test_loss_dict[epoch], label=legend[epoch])

fig.text(0.5, 0.04, 'Epoch', ha='center')
fig.text(0.04, 0.5, 'Loss', va='center', rotation='vertical')
ax1.set_title("Training data")
ax1.legend()
ax2.set_title("Testing data")
ax2.legend()

plt.show()

fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
min_list_train = []
for epoch in epochs:
    min_list_train.append(min(train_loss_dict[epoch]))
ax1.bar(legend, min_list_train)

min_list_test = []
for epoch in epochs:
    min_list_test.append(min(test_loss_dict[epoch]))
ax2.bar(legend, min_list_test)

fig.text(0.5, 0.04, 'Epoch', ha='center')
fig.text(0.04, 0.5, 'Loss', va='center', rotation='vertical')
ax1.set_title("Training data")
ax2.set_title("Testing data")

plt.show()
