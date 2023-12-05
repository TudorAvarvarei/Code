from tensorboard.backend.event_processing import event_accumulator
import os

# Specify the path to your TensorBoard log directory
log_dir = './runs/2023-11-20_neurons_1000_epochs_10'

# Find the latest event file in the log directory
event_file = max([os.path.join(log_dir, d) for d in os.listdir(log_dir) if d.startswith("events")])

# Create an EventAccumulator
event_acc = event_accumulator.EventAccumulator(event_file)

# Load the TensorBoard events
event_acc.Reload()

# Get the available keys
available_keys = event_acc.Tags()
print("Available keys:", available_keys)

# Access data associated with a specific key
if 'train_loss' in available_keys['scalars']:
    train_loss_data = event_acc.Scalars('train_loss')
    # train_accuracy_data = event_acc.Scalars('train_accuracy')
    test_loss_data = event_acc.Scalars('test_loss')
    # test_accuracy_data = event_acc.Scalars('test_accuracy')
    for event in train_loss_data:
        print(f"Step {event.step}: {event.value}")
    # for event in train_accuracy_data:
    #     print(f"Step {event.step}: {event.value}")
    for event in test_loss_data:
        print(f"Step {event.step}: {event.value}")
    # for event in test_accuracy_data:
    #     print(f"Step {event.step}: {event.value}")



# Access other data types (histograms, images, etc.) using relevant methods
# For example, event_acc.Histograms('your_histogram_key'), event_acc.Images('your_image_key'), etc.
