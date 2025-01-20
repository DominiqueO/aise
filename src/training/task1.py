"""Implementation of 1D Fourier Neural Operator
adapted from:
https://colab.research.google.com/drive/1dJS57eiof_ZVKm4tPgclNVHa_qP4gAih?usp=sharing#scrollTo=-dFB-dqncmmt
"""
import torch
import torch.nn as nn
import os
import numpy as np
from pip._internal import resolution
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt

from src.models.FNO import FNO1d
import src.training.Training as Train

data_path = "../../data/task1/"
torch.manual_seed(0)
np.random.seed(0)


n_train = 64 # number of training samples

# Load the data
y_data = torch.from_numpy(np.load(data_path + "train_sol.npy")).type(torch.float32)
# x_data is inferred from shape of y_data, assuming equal spacing of data points
x_data = torch.linspace(0, 1, y_data.shape[-1]).type(torch.float32).unsqueeze(0)

# Plot training data to check if data is loaded correctly
plt.figure()
plt.title('Set of trajectories')
plt.xlabel('x')
plt.ylabel('u(x,t)')
for i in range(5):
    plt.plot(x_data.squeeze().numpy(), y_data[np.random.randint(0, len(y_data) - 1), i], label="t=" + str(i/4))

plt.legend()
plt.savefig("../../deliverables/task1_1.pdf", format="pdf")
plt.show()

# Expand x_data to size of y_data (i.e. make identical copies of x for each time step t)
x_data_expanded = x_data.expand(y_data.size(0), -1)

# Set up the dataloaders for training and validation
input_function_train = torch.stack((y_data[:n_train, 0, :], x_data_expanded[:n_train]), dim=2).type(torch.float32)
output_function_train = y_data[:n_train, -1, :]
input_function_test = torch.stack((y_data[n_train:, 0, :], x_data_expanded[n_train:]), dim=2).type(torch.float32)
output_function_test = y_data[n_train:, -1, :]

batch_size = 10

training_set = DataLoader(TensorDataset(input_function_train, output_function_train), batch_size=batch_size, shuffle=True)
testing_set = DataLoader(TensorDataset(input_function_test, output_function_test), batch_size=batch_size, shuffle=False)

# Hyperparameters for training
learning_rate = 0.001
epochs = 250
step_size = 50
gamma = 0.5

# Initialize model
modes = 16
width = 64
fno = FNO1d(modes, width) # model

optimizer = Adam(fno.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
l = torch.nn.MSELoss()
freq_print = 1

fno, loss_history, relative_l2 = Train.train_NN(fno, optimizer, scheduler, l, training_set, testing_set, epochs,
                                                freq_print, save=False, save_name="1dFNO", squeeze_dim=2)





# Plot predicted function for visual check
fno.eval()
idx_data = 10
input_function_test_n = input_function_test[idx_data, :, :].unsqueeze(0)
output_function_test_n = output_function_test[idx_data, :].unsqueeze(0)
output_function_test_pred_n = fno(input_function_test_n)

fig = plt.figure()
plt.title('Solution for t=1.0 approximated by FNO')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.grid(True, which="both", ls=":")
plt.plot(input_function_test_n[0,:,1].detach(), output_function_test_n[0].detach(), label="True Solution", c="C0", lw=2)
plt.scatter(input_function_test_n[0,:,1].detach(), output_function_test_pred_n[0].detach(), label="Approximate Solution", s=8, c="C1")
plt.legend()
text = 'err={}'.format(relative_l2[-1])
plt.text(0.2,0.2, text, transform=fig.transFigure)
plt.savefig("../../deliverables/task1_2.pdf", format="pdf")
plt.show()

# Plot error history and loss
fig = plt.figure()
plt.title('Epochs and L2 Error')
plt.xlabel('epochs')
plt.ylabel('L2 Error')
plt.grid(True, which="both", ls=":")
plt.plot(range(epochs), relative_l2, label="L2 Error")
plt.legend()
plt.savefig("../../deliverables/task1_3.pdf", format="pdf")
plt.show()

# Plot loss history
fig = plt.figure()
plt.title('Epochs and Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid(True, which="both", ls=":")
plt.plot(range(epochs), loss_history, label="Loss")
plt.legend()
plt.savefig("../../deliverables/task1_4.pdf", format="pdf")
plt.show()


# %% Subtask 2

res_list = range(32, 129, 32)

# Load the data for different resolutions
plt.figure()
plt.title('L2 Error at different spatial resolutions')
plt.xlabel('resolution (number of points)')
plt.ylabel('error')

for res in res_list:
    y_data = torch.from_numpy(np.load(data_path + "test_sol_res_{}.npy".format(res))).type(torch.float32)
    # x_data is inferred from shape of y_data, assuming equal spacing of data points
    x_data = torch.linspace(0, 1, y_data.shape[-1]).type(torch.float32).unsqueeze(0)
    # Expand x_data to size of y_data (i.e. make identical copies of x for each time step t)
    x_data_expanded = x_data.expand(y_data.size(0), -1)
    # Use data at time t=0 to predict data at time t=0
    input_function = torch.stack((y_data[:, 0, :], x_data_expanded), dim=2).type(torch.float32)
    output_function = y_data[:, -1, :] # data at time t=1 as target
    with torch.no_grad():
        output_function_pred = fno(input_function).squeeze(2)

    l2_error = (torch.norm(output_function_pred - output_function, dim=1) / torch.norm(output_function, dim=1)).mean().item()
    print("Average l2 error for resolution r={}: {}".format(res, l2_error))

    plt.plot(res, l2_error, 'o', label='error at resolution r={}'.format(res))

plt.legend()
plt.savefig("../../deliverables/task1_5.pdf", format="pdf")
plt.show()
