

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import src.training.Training as Train
from src.models.FuncApprox import FuncApprox
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset



# Assume u is a (N, T) array, x and t are (N, T) arrays
# dx and dt are the grid spacings for x and t, respectively

def compute_derivatives(u, dx, dt):
    # First derivatives
    u_x = np.zeros_like(u)
    u_t = np.zeros_like(u)

    # Compute u_x using central differences
    u_x[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
    u_x[0, :] = (u[1, :] - u[0, :]) / dx  # Forward difference at the start
    u_x[-1, :] = (u[-1, :] - u[-2, :]) / dx  # Backward difference at the end

    # Compute u_t using central differences
    u_t[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dt)
    u_t[:, 0] = (u[:, 1] - u[:, 0]) / dt  # Forward difference at the start
    u_t[:, -1] = (u[:, -1] - u[:, -2]) / dt  # Backward difference at the end

    # Second derivatives
    u_xx = np.zeros_like(u)
    u_tt = np.zeros_like(u)

    # Compute u_xx using central differences
    u_xx[1:-1, :] = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / (dx ** 2)
    u_xx[0, :] = u_xx[1, :]  # Approximation for boundary points
    u_xx[-1, :] = u_xx[-2, :]  # Approximation for boundary points

    # Compute u_tt using central differences
    u_tt[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / (dt ** 2)
    u_tt[:, 0] = u_tt[:, 1]  # Approximation for boundary points
    u_tt[:, -1] = u_tt[:, -2]  # Approximation for boundary points

    return u_x, u_t, u_xx, u_tt




if __name__ == "__main__":
    # Paths
    data_path = '../data/task2/' # for data
    figure_path = '../deliverables/' # for figures

    # Load data
    with np.load(data_path + '1.npz') as data:
        x_data = data['x']
        t_data = data['t']
        u_data = data['u']
        N, T = x_data.shape # N=spatial resolution, T=temporal resolution

    # Visualise trajectories
    norm = mcolors.Normalize(vmin=t_data[0, 0], vmax=t_data[0, -1])
    colormap = cm.inferno
    plt.figure()
    plt.title('Trajectories for different times t')
    plt.grid(True, which="both", ls=":")
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    for t in range(T):
        if t_data[0, t] % 1 == 0:
            label = 'time=' + str(t_data[0, t])
        else:
            label = None
        color = colormap(norm(t_data[0, t]))
        plt.plot(x_data[:, t], u_data[:, t], color=color, label=label, linewidth=1)
    plt.legend()
    plt.savefig(figure_path + 'task2_1.pdf', format='pdf')
    plt.show()


    # Preprocess data
    # 1. Flatten arrays and combine x and t
    # 2. Reshape input array such that each (x, t) pair forms its own subarray
    # 3. Choose random (x, y) pairs with corresponding u for training
    input = np.stack([x_data.ravel(), t_data.ravel()], axis=1)
    input = np.reshape(input, (N * T, -1))
    output = np.reshape(u_data, (N * T, -1))
    # Size of training set
    n_train = int(0.7 * N * T)
    training_indices = np.random.choice(len(input), size=n_train, replace=False)
    input_function_train = input[training_indices]
    output_function_train = output[training_indices].squeeze()
    input_function_test = input[~training_indices]
    output_function_test = output[~training_indices].squeeze()

    batch_size = 16

    # Set up the dataloaders for training and validation
    input_function_train = torch.from_numpy(input_function_train).float()
    output_function_train = torch.from_numpy(output_function_train).float()
    input_function_test = torch.from_numpy(input_function_test).float()
    output_function_test = torch.from_numpy(output_function_test).float()

    training_set = DataLoader(TensorDataset(input_function_train, output_function_train), batch_size=batch_size,
                              shuffle=True)
    testing_set = DataLoader(TensorDataset(input_function_test, output_function_test), batch_size=batch_size,
                             shuffle=False)

    # Hyperparameters
    learning_rate = 0.003
    epochs = 10
    step_size = 25
    gamma = 0.5

    # Initialize model
    model = FuncApprox(2, 1, 64)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss = torch.nn.MSELoss()

    function_model, losses, l2_error = Train.train_NN(model, optimizer, scheduler, loss, training_set, testing_set,
                                                      epochs=epochs, squeeze_dim=1)

    # Plot predicted function for visual check (random points)
    function_model.eval()
    idx_data = 3
    input_function_test_n = input_function_test[0:30]
    output_function_test_n = output_function_test[0:30]
    output_function_test_pred_n = function_model(input_function_test_n)

    plt.figure()
    plt.title('Approximation for arbitrary times t')
    plt.grid(True, which="both", ls=":")
    plt.scatter(input_function_test_n[:, 0].detach(), output_function_test_n.detach(), label="True Solution", c="C0",
             lw=2)
    plt.scatter(input_function_test_n[:, 0].detach(), output_function_test_pred_n.detach(),
                label="Approximate Solution", s=8, c="C1")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.savefig(figure_path + 'task2_2.pdf', format='pdf')
    plt.show()

    # Plot predicted function for visual check (fixed time t)
    t = 0.0
    indices = torch.nonzero(input_function_test[:, 1] == t, as_tuple=False).ravel()
    input_function_test_n = input_function_test[indices, :]
    output_function_test_n = output_function_test[indices]
    output_function_test_pred_n = function_model(input_function_test_n)

    plt.figure()
    plt.title('Approximation for time t='+str(t))
    plt.grid(True, which="both", ls=":")
    plt.plot(input_function_test_n[:, 0].detach(), output_function_test_n.detach(), label="True Solution", c="C0",
             lw=2)
    plt.scatter(input_function_test_n[:, 0].detach(), output_function_test_pred_n.detach(),
                label="Approximate Solution", s=8, c="C1")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.savefig(figure_path + 'task2_3.pdf', format='pdf')
    plt.show()




