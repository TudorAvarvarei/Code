from tensorboard.backend.event_processing import event_accumulator
from matplotlib import pyplot as plt
import os
import numpy as np

# Specify the path to your TensorBoard log directory
log_dirs = ['./runs/2024-03-0150_refined',
            './runs/2024-03-0175_refined',
            './runs/2024-03-01100_refined',
            './runs/2024-03-01150_refined',
            './runs/2024-03-01200_refined',
            './runs/2024-03-01300_refined',
            './runs/2024-03-01400_refined',
            './runs/2024-03-01500_refined']
            # './runs/2024-02-15_neurons_600_epochs_10',
            # './runs/2024-02-15_neurons_700_epochs_10',
            # './runs/2024-02-15_neurons_800_epochs_10',
            # './runs/2024-02-15_neurons_1000_epochs_10',]

# log_dirs_high_thresh = ['./runs/2024-02-21thresh_5_leaks_[-2.0, 0]',
#                         './runs/2024-02-21thresh_5_leaks_[-2.0, 0.5]',
#                         './runs/2024-02-21thresh_5_leaks_[-2.0, 1.5]',
#                         './runs/2024-02-21thresh_5_leaks_[0.0, 0.0]',
#                         './runs/2024-02-21thresh_5_leaks_[0.0, 0.5]',
#                         './runs/2024-02-21thresh_5_leaks_[0.0, 1.5]',
#                         './runs/2024-02-21thresh_5_leaks_[2.0, 0.5]',
#                         './runs/2024-02-21thresh_5_leaks_[2.0, 1.5]']


epochs = [50, 75, 100, 150, 200, 300, 400, 500]
# legend = ["Leak\nmean=-2\nSTD=0",
#           "Leak\nmean=-2\nSTD=0.5",
#           "Leak\nmean=-2\nSTD=1.5",
#           "Leak\nmean=0\nSTD=0",
#           "Leak\nmean=0\nSTD=0.5",
#           "Leak\nmean=0\nSTD=1.5",
#           "Leak\nmean=2\nSTD=0.5",
#           "Leak\nmean=2\nSTD=1.5"]

log_dirs = ["./runs/2024-04-10_controller_e2e"]
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

print(train_loss_dict)
print(test_loss_dict)

# train_loss_dict_high_thresh = {}
# test_loss_dict_high_thresh = {}
# i=0
# 
# for log_dir in log_dirs_high_thresh:
#     train_loss_dict_high_thresh[epochs[i]]=[]
#     test_loss_dict_high_thresh[epochs[i]]=[]
#     # Find the latest event file in the log directory
#     event_file = max([os.path.join(log_dir, d) for d in os.listdir(log_dir) if d.startswith("events")])

#     # Create an EventAccumulator
#     event_acc = event_accumulator.EventAccumulator(event_file)

#     # Load the TensorBoard events
#     event_acc.Reload()

#     # Get the available keys
#     available_keys = event_acc.Tags()

#     # Access data associated with a specific key
#     if 'train_loss' in available_keys['scalars']:
#         train_loss_data = event_acc.Scalars('train_loss')
#         test_loss_data = event_acc.Scalars('test_loss')
#         for event in train_loss_data:
#             train_loss_dict_high_thresh[epochs[i]].append(event.value)
#         for event in test_loss_data:
#             test_loss_dict_high_thresh[epochs[i]].append(event.value)

#     i += 1

# print("train_loss_dict:", train_loss_dict)
# print("test_loss_dict:", test_loss_dict)

# fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
# for epoch in epochs:
#     ax1.plot(train_loss_dict[epoch], label=legend[epoch])

# for epoch in epochs:
#     ax2.plot(test_loss_dict[epoch], label=legend[epoch])

# fig.text(0.5, 0.04, 'Epoch', ha='center')
# fig.text(0.04, 0.5, 'Loss', va='center', rotation='vertical')
# ax1.set_title("Training data")
# ax1.legend()
# ax2.set_title("Testing data")
# ax2.legend()

# plt.show()

# fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
# min_list_train = []
# for epoch in epochs:
#     min_list_train.append(min(train_loss_dict[epoch]))
# ax1.bar(legend, min_list_train)

min_list_test = []
for epoch in epochs:
    min_list_test.append(min(train_loss_dict[epoch]))#/len(train_loss_dict[epoch]))

plt.plot(epochs, min_list_test, linewidth=2)
plt.xlabel("No of spiking neurons", fontdict={"size":15})
plt.ylabel("MSE Loss", fontdict={"size":15})
plt.title("Learning performance for varying number of spiking neurons", fontdict={"size":15})
plt.show()
# min_list_test_high_thresh = []
# for epoch in epochs:
#     min_list_test_high_thresh.append(min(test_loss_dict_high_thresh[epoch]))

# final_dictionary = {"Threshold; Mean=1; STD=0.5": min_list_test, "Threshold; Mean=5; STD=0.5": min_list_test_high_thresh}


# # x = np.arange(len(species))  # the label locations
# width = 0.35  # the width of the bars
# multiplier = 0

# fig, ax = plt.subplots()
# epochs = np.array(epochs)

# for attribute, measurement in final_dictionary.items():
#     offset = width * multiplier
#     rects = ax.bar(epochs + offset, measurement, width, label=attribute)
#     multiplier += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('MSE Loss', fontdict={"size":16})
# ax.set_title("Learning performance with various threshold and leak initial values", fontdict={"size":20})
# ax.set_xticks(epochs + (width / 2), legend, fontsize=15)
# ax.legend(loc='best', ncols=3, fontsize=12)

# # ax.set_ylim(0, 250)

# plt.show()
# plt.bar(legend, min_list_test)
# plt.title("Learning performance with various threshold and leak initial values", fontdict={"size":15})
# plt.ylabel('MSE Loss', fontdict={"size":15})
# plt.rcParams["font.size"] = 15

# plt.text(0.04, 0.5, 'MSE Loss', va='center', rotation='vertical', fontdict={"size":15})
# # ax1.set_title("Training data")
# # ax2.set_title("Testing data")

# plt.show()

{'u1': 0.15894058, 'u2': 0.071065664, 'u3': 0.10074704, 'u4': 0.106414564}
{'u1': 0.1719831, 'u2': 0.079242915, 'u3': 0.10887638, 'u4': 0.1091249}
{'u1': 0.15341207, 'u2': 0.06952137, 'u3': 0.09299672, 'u4': 0.09814947}
