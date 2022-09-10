from typing import Dict, Union, List

import numpy as np
import scipy as sci
from numpy import ndarray

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from scipy import integrate
from scipy.special import ellipj, ellipk
from ODE_Jacs import *
# from ODE_Jacs_np import *
import os

BATCH_SIZE = 50
WEIGHT_DECAY = 0
LEARNING_RATE = 5e-3
NUMBER_EPOCHS = 500
IN_DIM = 4
OUT_DIM = 2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seed(seed=0):
    """Set one seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_driven_data(tmax=20, dt=1, theta0=2.0, d_theta0=0.0, A=1.0, k=1.0, b=0.5, g=9.81):
    t_lim = (0, tmax)
    t = np.arange(0, tmax, dt)
    omega_0 = np.sqrt(g)
    y0 = np.array([theta0, d_theta0])
    A_sol = np.ones_like(t) * A
    sol = integrate.solve_ivp(pendulum, t_lim, y0, t_eval=t, args=(A, k, b, omega_0))
    return np.hstack([sol['y'].T, k * np.expand_dims(sol['t'], axis=1),
                      np.expand_dims(A_sol, axis=1)])


def pendulum(t, y, A=1.0, k=1.0, b=0.1, omega_0=np.sqrt(9.81)):
    theta = y[0]  # - np.pi*(np.abs(y[0])//np.pi)*np.sign(y[0])
    dtheta = y[1]
    ddtheta = -omega_0 ** 2 * np.sin(theta) + (-b * dtheta + A * np.cos(k * t))
    return [dtheta, ddtheta]


def create_data(tmax=20, dt=1, theta0=2.0, d_theta0=0.0):
    """Solution for the nonlinear pendulum in theta space."""
    t = np.arange(0, tmax, dt)
    g = 9.81
    alpha = np.arccos(np.cos(theta0) - 0.5 * d_theta0 ** 2 / g)
    S = np.sin(0.5 * alpha)
    K_S = ellipk(S ** 2)
    omega_0 = np.sqrt(g)
    sn, cn, dn, ph = ellipj(K_S - omega_0 * t, S ** 2)
    theta = 2.0 * np.arcsin(S * sn)
    d_sn_du = cn * dn
    d_sn_dt = -omega_0 * d_sn_du
    d_theta_dt = 2.0 * S * d_sn_dt / np.sqrt(1.0 - (S * sn) ** 2)
    A_sol = np.ones_like(t) * A
    return np.stack([theta, d_theta_dt, t, A_sol], axis=1)


def create_dataloader(x, x_test=None, batch_size=BATCH_SIZE, shuffle=True):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(np.asarray(x[0:-1, :IN_DIM]), dtype=torch.double).to(device),
        torch.tensor(np.asarray(x[1:, :OUT_DIM]), dtype=torch.double).to(device),
    )

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

    if x_test is None:
        x_test = x
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(np.asarray(x_test[0:-1, :IN_DIM]), dtype=torch.double).to(device),
        torch.tensor(np.asarray(x_test[1::, :OUT_DIM]), dtype=torch.double).to(device),
    )

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def euler_step_func(f, x, dt, monitor=False):
    """The 'forward' Euler, a one stage Runge Kutta."""
    k1 = f(x)
    x_out = x + dt * k1
    if monitor:
        return x_out, k1
    else:
        return x_out


def rk4_step_func(f, x, dt, monitor=False):
    """The 'classic' RK4, a four stage Runge Kutta, O(Dt^4)."""
    k1 = f(x)
    x1 = x + 0.5 * dt * k1
    k2 = f(x1)
    x2 = x + 0.5 * dt * k2
    k3 = f(x2)
    x3 = x + dt * k3
    k4 = f(x3)
    x_out = x + dt * (1.0 / 6.0 * k1 + 1.0 / 3.0 * k2 + 1.0 / 3.0 * k3 +
                      1.0 / 6.0 * k4)
    if monitor:
        return x_out, torch.vstack([k1, k2, k3, k4])
    else:
        return x_out


def rk4_38_step_func(f, x, dt, monitor=False):
    """"The alternate '3/8' RK4, a four-stage Runge Kutta with different coefficients """
    k1 = f(x)
    x1 = x + 1.0 / 3.0 * dt * k1
    k2 = f(x1)
    x2 = x + (-1.0 / 3.0 * k1 + k2) * dt
    k3 = f(x2)
    x3 = x + (k1 - k2 + k3) * dt
    k4 = f(x3)
    x_out = x + dt * 1.0 / 8.0 * (k1 + 3 * k2 + 3 * k3 + k4)
    if monitor:
        return x_out, torch.vstack([k1, k2, k3, k4])
    else:
        return x_out


def shallow(in_dim, hidden, out_dim, layers=1, Act=torch.nn.Tanh):
    """Just make a shallow network. This is more of a macro."""
    modules = [torch.nn.Linear(in_dim, hidden), Act()]
    for i in range(layers - 1):
        modules.append(torch.nn.Linear(hidden, hidden))
        modules.append(Act())
    modules.append(torch.nn.Linear(hidden, out_dim))
    return torch.nn.Sequential(*modules)


class ShallowODE(torch.nn.Module):
    """A basic shallow network that takes in a t as well"""

    def __init__(self, in_dim, out_dim, hidden=10, layers=1, Act=torch.nn.Tanh):
        super(ShallowODE, self).__init__()
        self.net = shallow(in_dim, hidden, out_dim, layers, Act=Act)

    def forward(self, x, h, dt, method='euler'):
        if method == 'euler':
            # print(method)
            # for i in range(int(dt/h)):
            x = euler_step_func(self.net, x, dt)
            return x
        elif method == 'rk4':
            # print(method)
            # for i in range(int(dt/h)):
            x = rk4_step_func(self.net, x, dt)
            return x
        elif method == 'rk4_38':
            # print(method)
            # for i in range(int(dt/h)):
            x = rk4_38_step_func(self.net, x, dt)
            return x


def validate(ODEnet, test_loader, method='rk4', dt=0.0001, h=0.1):
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            outputs = ODEnet(inputs, h=dt, dt=dt, method=method)
            loss = criterion(outputs[:, :OUT_DIM], targets)
            epoch_loss += loss.item()
    return epoch_loss / batch_idx


def train(ODEnet, train_loader, test_loader=None, lr=LEARNING_RATE, wd=WEIGHT_DECAY, method='rk4',
          dt=0.0001, loss_list=[], h=0.1, best_loss=100, fig_name=None, save_epochs=100):
    optimizer_ODEnet = optim.Adam(ODEnet.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_steps = int(h / dt)
    if test_loader is not None:
        val_losses = []
        val_epochs = []
    criterion = torch.nn.MSELoss()
    best_state = ODEnet.net.state_dict()
    ode_loss_hist = loss_list
    epoch_losses = []
    print('ODENet Training')
    for epoch in range(0, NUMBER_EPOCHS):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer_ODEnet.zero_grad()
            outputs = ODEnet(inputs, h=dt, dt=dt, method=method)
            loss = criterion(outputs[:, :OUT_DIM], targets)
            loss.backward()
            epoch_loss += loss.item()
            optimizer_ODEnet.step()
            ode_loss_hist.append(loss.item())
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = ODEnet.net.state_dict()

        epoch_losses.append(epoch_loss / batch_idx)
        if epoch % 50 == 0:
            print(f'Epoch: {epoch}, Training Loss: {epoch_loss / batch_idx}')
            if test_loader is not None:
                plt.figure(17)
                val_loss = validate(ODEnet, test_loader, dt=dt, h=h)
                val_losses.append(val_loss)
                val_epochs.append(epoch)
                if epoch >0:
                    plt.plot(val_epochs, val_losses)
                    plt.title(f'{integrator} Model Validation Losses')
                    plt.xlabel('Epoch')
                    plt.ylabel('Val Loss')
                    plt.yscale('log')
                    plt.xlim([0, NUMBER_EPOCHS])
                    plt.savefig(fig_name.replace('Train', 'Val'))
                    plt.close()
                print(f'Val Loss: {val_loss}')
            if fig_name is not None and epoch>0:
                plt.figure(15)
                plt.plot(range(epoch+1), epoch_losses)
                plt.title(f'{integrator} Model Losses')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.yscale('log')
                plt.xlim([0, NUMBER_EPOCHS])
                plt.savefig(fig_name)
                plt.close()
    ODEnet.net.load_state_dict(best_state)
    return ODEnet, epoch_losses, best_loss


def generate_datasets(A, b, theta_0s, dt, train_steps, g=9.81, k=1):
    x_list = []
    for theta_0i in theta_0s:
        xi = create_driven_data(tmax=T_MAX, dt=dt, theta0=theta_0i, A=A, b=b, g=g, k=k)
        n_batches = xi.shape[0] // (BATCH_SIZE * train_steps)
        x_list.append(xi[:BATCH_SIZE * train_steps * n_batches])
    x_set = np.vstack(x_list)
    return x_set


def find_error(ODEnet, A, theta_0i, b=1, k=1, g=1, h=0.1, dt=0.1, shuffle=True):
    steps = int(h / dt)
    target_list = []
    output_list = []
    diff_list = []
    x_test = create_driven_data(tmax=T_MAX, dt=dt, theta0=theta_0i, A=A, b=b, g=g, k=k)
    n_batches = x_test.shape[0] // (BATCH_SIZE * train_steps)
    x_test = x_test[:n_batches * BATCH_SIZE * train_steps, :IN_DIM]
    train_loader, test_loader = create_dataloader(x_train_full[:1], x_test[::steps],
                                                  batch_size=BATCH_SIZE, shuffle=shuffle)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        outputs = ODEnet(inputs[:, :IN_DIM], h=h, dt=h, method=integrator)
        output_list.append(outputs.detach().cpu().numpy()[:, :OUT_DIM])
        target_list.append(targets[:, :OUT_DIM].cpu().numpy())
    error = np.mean(np.linalg.norm(np.vstack(output_list) - np.vstack(target_list),
                                   axis=1) ** 2)
    return error


if __name__ == '__main__':
    dt = 0.0001
    train_h = 0.1
    omega_0 = np.sqrt(9.81)
    N_points = 50000
    LE_points = 500000
    T_MAX = N_points * dt
    set_seed(5544)
    folder = 'Driven'
    integrators = ['euler', 'rk4']
    colors = {'euler': 'seagreen', 'rk4': 'purple'}
    plot = False
    train_models = True
    evaluate = True
    # Set System Parameters:
    hidden = 100  # Neural ODE hidden size
    layers = 8
    k = 1.0  # Driving Force Frequency
    b = 1.0  # Damping coefficient
    # A = 0         # Driving force magnitude
    g = 1.0  # Acceleration due to gravity
    train_min = 3  # Minimum A used for training
    train_max = 10  # Maximum A used for training

    A_train_name = f'{train_min}_{train_max}'
    A_test_name = f'all'
    suffix = f'l{layers}_{A_train_name}_'

    if plot:
        n_points = 20
        cloud_size = 0.2
        theta_0 = 0.5
        d_theta_0 = 0.2
        theta_0s, d_theta_0s = theta_0 + cloud_size * (np.random.rand(n_points) - 0.5), d_theta_0 + cloud_size * (
                np.random.rand(n_points) - 0.5)
        e_steps = 10

        # Generate Trajectories of different lengths and plot in phase space
        # A_plot = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        A_plot = []
        for A in A_plot:
            for e_steps in [1000000]:
                # Plot Evolution of different initial conditions
                fig = plt.figure(figsize=(8, 6))
                circle1 = plt.Circle((theta_0, d_theta_0), cloud_size, color='k', alpha=0.2)
                for idx, (theta_0i, d_theta_0i) in enumerate(zip(theta_0s, d_theta_0s)):
                    x = create_driven_data(tmax=T_MAX * 10, dt=dt, theta0=theta_0i, d_theta0=d_theta_0i, A=A, k=k,
                                           b=b)
                    i_x = np.where((np.abs(x[:, 0] - theta_0i) < 10e-3) & (np.abs(x[:, 1] - d_theta_0i) < 10e-3))[0]
                    if i_x.shape[0] == 0:
                        continue
                    elif i_x[0] + e_steps < x.shape[0]:
                        x_offsets = np.round(x / np.pi) * np.pi * np.repeat(np.array([[1., 0., 0., 0.]]), x.shape[0],
                                                                            axis=0)
                        x_plot = (x - x_offsets)[i_x[0]:i_x[0] + e_steps]
                        plt.plot(x_plot[:, 0], x_plot[:, 1], alpha=0.1)
                        xs = np.stack([x_plot[0], x_plot[-1]])
                        plt.scatter(xs[:, 0], xs[:, 1], s=8, c='k')
                # plt.legend()
                fig = plt.gcf()
                ax = fig.gca()
                ax.add_patch(circle1)
                print(f'Printing Trajectories for A = {A}, {e_steps} steps')
                plt.title(f'Evolution for A = {A}, {e_steps} Steps')
                plt.xlabel(r'$\theta$')
                plt.ylabel(r'$\frac{d\theta}{dt}$')
                plt.savefig(f'{folder}/Figures/Clouds/cloud_steps_driven{A}_{e_steps}.png')
                plt.close()

    # Create training and testing data
    xs = []
    n_theta_train = 50
    n_theta_test = 15
    theta_0_max = np.pi / 3
    theta_train = np.random.uniform(low=0, high=theta_0_max, size=n_theta_train)
    theta_test = np.random.uniform(low=0, high=theta_0_max, size=n_theta_test)
    # theta_train = [0.1, 0.3, 0.5, 1.0, 1.2, 1.5, 2.0, 2.2, 2.5]
    # theta_test = [0.6, 1.3, 2.3]

    A_list_train = {'low': [0.1, 0.8, 1.6, 2.4, 3.2, 3.8, 4.0],
                    'mid': [4.0, 4.8, 5.6, 6.4, 7.2, 8.2, 8.9],
                    'high': [9.0, 9.8, 10.6, 11.4, 12.2, 13.1, 13.8, 14.6, 14.9, 15.8, 16.6]}

    b_list_train = [b]

    A_list_test = {'low': [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.8],
                   'mid': [4.2, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.8],
                   'high': [9.2, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0]}

    all_list_train = []
    for a in A_list_train:
        # print(a)
        all_list_train = all_list_train + A_list_train[a]
    all_list_test = []
    for a in A_list_test:
        all_list_test = all_list_test + A_list_test[a]
    A_list_train['all'] = np.array(all_list_train)
    A_list_test['all'] = np.array(all_list_test)
    A_list_test[A_test_name] = A_list_test['all']
    b_list_test = [b]
    train_steps = int(train_h / dt)

    A_list_train[A_train_name] = A_list_train['all'][(A_list_train['all'] <= train_max) *
                                                     (A_list_train['all'] >= train_min)]

    # Generate Train and Test Data
    for A in A_list_train[A_train_name]:
        for b in b_list_train:
            x_train_full = []
            x_test_full = []
            fname = f'{folder}/Data/AltData_A{A}_b{b}.p'
            if not os.path.exists(f'{folder}/Data/AltData_A{A}_b{b}.p'):
                print('Generating Data')
                x_train = generate_datasets(A, b, theta_train, dt, train_steps)
                x_test = generate_datasets(A, b, theta_test, dt, train_steps)
                torch.save((x_train, x_test), fname)
            else:
                print('Loading Data')
                x_train, x_test = torch.load(fname)
            x_train_full.append(x_train)
            x_test_full.append(x_test)
        if plot:
            plt.figure(100)
            plt.plot(x_train[:, 0], x_train[:, 1], label=A)
    x_train_full = np.vstack(x_train_full)
    x_test_full = np.vstack(x_test_full)

    plt.legend()
    plt.xlabel(r'$\theta$')
    # plt.xlim([-3, 3])
    plt.ylabel(r'$\frac{d\theta}{dt}$')
    # plt.ylim([-6, 6])
    plt.title('Train Set Trajectories')
    plt.savefig(f'{folder}/Figures/Trajectories/Train_{A_train_name}.png')

    # Train ODEnets
    if train_models:
        train_loader, test_loader = create_dataloader(x_train_full[::train_steps, :IN_DIM],
                                                      x_test_full[::train_steps, :IN_DIM],
                                                      batch_size=BATCH_SIZE, shuffle=True)
        for integrator in ['euler', 'rk4']:
            model_name = f'{folder}/Models/model_{suffix}{integrator}.p'
            if os.path.isfile(model_name):
                print(f'Loading {integrator} Model')
                ODEnet, ode_loss_hist, best_loss = torch.load(model_name, map_location=device)
                ODEnet, ode_loss_hist, best_loss = train(ODEnet, train_loader, method=integrator, dt=train_h,
                                                         loss_list=ode_loss_hist, best_loss=best_loss, h=dt)
                ODEnet = ODEnet.double().to(device)
                torch.save((ODEnet, ode_loss_hist, best_loss), model_name)
            else:
                ODEnet = ShallowODE(in_dim=IN_DIM, hidden=hidden, layers=layers, out_dim=IN_DIM,
                                    Act=torch.nn.Tanh).double().to(device)
                ODEnet, ode_loss_hist, best_loss = train(ODEnet, train_loader, test_loader=test_loader,
                                                         method=integrator, dt=train_h, h=dt,
                                                         fig_name=f'{folder}/Train Losses/loss_plot_{integrator}_{A_train_name}.png')
                torch.save((ODEnet, ode_loss_hist, best_loss), model_name)

    if evaluate:
        A_test_name = 'all'
        h = train_h
        # Redo plotting to have heatmaps of loss (color) vs. A (x) and h (y)
        # Increase number of layers to 4 or 8 and see if it learns better
        train_loader, _ = create_dataloader(x_train_full[::train_steps, :IN_DIM],
                                            x_test_full[::train_steps, :IN_DIM],
                                            batch_size=BATCH_SIZE)
        with torch.no_grad():
            models = {}
            # Evaluate the error of the models
            error_means = {}
            error_stds = {}
            train_error_means = {}
            train_error_stds = {}
            A_test = A_list_test[A_test_name]
            for integrator in integrators:
                model_name = f'{folder}/Models/model_{suffix}{integrator}.p'
                ODEnet, loss_list, _ = torch.load(model_name, map_location=device)
                models[f'{integrator}{suffix}'] = ODEnet
                all_errors = []
                train_errors = []

                # Calculate Error for thetas in test set for each A
                for theta_0i in theta_test:
                    error = []
                    for A in A_list_test[A_test_name]:
                        # print(f'A = {A}, Integrator {integrator}, theta = {theta_0i}')
                        error.append(find_error(ODEnet, A=A, theta_0i=theta_0i, h=h, dt=dt, k=k, b=b, g=g,
                                                shuffle=False))
                    error = np.hstack(error)
                    all_errors.append(error)
                for theta_0i in theta_train:
                    error_train = []
                    for A in A_list_test[A_test_name]:
                        error_train.append(find_error(ODEnet, A=A, theta_0i=theta_0i, h=h, dt=dt, k=k, b=b, g=g,
                                                      shuffle=False))
                    error_train = np.hstack(error_train)
                    train_errors.append(error_train)
                all_errors = np.vstack(all_errors)
                train_errors = np.vstack(train_errors)

                # Calculate Error Statistics across thetas
                train_error_means[integrator] = np.mean(train_errors, axis=0)
                train_error_stds[integrator] = np.std(train_errors, axis=0)
                error_means[integrator] = np.mean(all_errors, axis=0)
                error_stds[integrator] = np.std(all_errors, axis=0)
                print(f'Plotting Errors for {integrator}, model {suffix}')
                y = error_means[integrator]
                std = error_stds[integrator]
                y_train = train_error_means[integrator]
                std_train = train_error_stds[integrator]
                if A_test_name not in ['low', 'mid', 'high']:
                    plt.axvline(x=train_min, color='k', linestyle='--')
                    plt.axvline(x=train_max, color='k', linestyle='--')
                plt.figure(1)
                plt.plot(A_test, y, 'o--', label=f'{integrator} Test', color=colors[integrator])
                plt.fill_between(A_test, y - std, y + std, alpha=0.5, color=colors[integrator],
                                 linewidth=0.0)
                plt.plot(A_test, y_train, 'o--', label=f'{integrator} Train', alpha=0.7,
                         color=f'medium{colors[integrator]}')
                plt.fill_between(A_test, y_train - std_train, y_train + std_train, alpha=0.3,
                                 color=f'medium{colors[integrator]}', linewidth=0.0)

            plt.figure(1)
            plt.yscale('log')
            plt.xlabel('A')
            plt.ylabel('Mean-square Error')
            plt.legend(fontsize=14)
            plt.title(f'Error for Models trained from A = {train_min} to {train_max}')
            plt.tight_layout()
            plt.savefig(f'{folder}/Figures/Errors/Models_errors_{suffix}vsH_test{A_test_name}.png')
            plt.close()
