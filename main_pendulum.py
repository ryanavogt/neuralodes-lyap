import numpy as np
import scipy as sci

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
NUMBER_EPOCHS = 1000
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
    A_sol = np.ones_like(t)*A
    sol = integrate.solve_ivp(pendulum, t_lim, y0, t_eval=t, args=(A, k, b, omega_0))
    return np.hstack([sol['y'].T, k*np.expand_dims(sol['t'], axis=1),
                      np.expand_dims(A_sol, axis=1)])


def pendulum(t, y, A=1.0, k=1.0, b=0.1, omega_0=np.sqrt(9.81)):
    theta = y[0]
    dtheta = y[1]
    ddtheta = -omega_0**2 * np.sin(theta) + (-b * dtheta + A * np.cos(k * t))
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


def create_dataloader(x, x_test=None, batch_size=BATCH_SIZE):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(np.asarray(x[0:-1, :IN_DIM]), dtype=torch.double).to(device),
        torch.tensor(np.asarray(x[1:, :OUT_DIM]), dtype=torch.double).to(device),
    )

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

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


def shallow(in_dim, hidden, out_dim, Act=torch.nn.Tanh):
    """Just make a shallow network. This is more of a macro."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        Act(),
        torch.nn.Linear(hidden, out_dim), )


class ShallowODE(torch.nn.Module):
    """A basic shallow network that takes in a t as well"""

    def __init__(self, in_dim, out_dim, hidden=10, Act=torch.nn.Tanh):
        super(ShallowODE, self).__init__()
        self.net = shallow(in_dim, hidden, out_dim, Act=Act)

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


def train(ODEnet, train_loader, lr=LEARNING_RATE, wd=WEIGHT_DECAY, method='rk4', dt=0.0001, loss_list=[],
          h=0.1, best_loss=100, fig_name=None):
    optimizer_ODEnet = optim.Adam(ODEnet.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_steps = int(h/dt)
    criterion = torch.nn.MSELoss()
    best_state = ODEnet.net.state_dict()
    ode_loss_hist = loss_list
    epoch_losses = []
    print('ODENet Training')
    for epoch in range(1, NUMBER_EPOCHS):
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

        epoch_losses.append(epoch_loss/batch_idx)
        if epoch % 50 == 0:
            print(f'Epoch: {epoch}, Loss: {epoch_loss/batch_idx}')
            if fig_name is not None:
                plt.figure(15)
                plt.plot(range(epoch), epoch_losses)
                plt.title(f'{integrator} Model Losses')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.yscale('log')
                plt.xlim([0, NUMBER_EPOCHS])
                plt.savefig(fig_name)
                plt.close()
    ODEnet.net.load_state_dict(best_state)
    return ODEnet, epoch_losses, best_loss


def generate_datasets(file_name, A, b, theta_0s, dt, train_steps, g=9.81, k=1):
    x_full = []
    x_list = []
    for theta_0i in theta_0s:
        xi = create_driven_data(tmax=T_MAX, dt=dt, theta0=theta_0i, A=A, b=b, g=g, k=k)
        n_batches = xi.shape[0] // (BATCH_SIZE * train_steps)
        x_list.append(xi[:BATCH_SIZE * train_steps * n_batches])
    x_set = np.vstack(x_list)
    return x_set


if __name__ == '__main__':
    dt = 0.0001
    train_h = 0.1
    omega_0 = np.sqrt(9.81)
    N_points = 500000
    LE_points = 500000
    T_MAX = N_points * dt
    set_seed(5544)
    folder = 'Driven'
    integrators = ['euler', 'rk4']
    plot = False
    train_models = False
    evaluate = True
    test_traj = False
    check_losses = False
    k = 1
    b = 0
    A = 0
    g = 9.81

    if plot:
        n_points = 20
        cloud_size = 0.2
        theta_0 = 0.5
        d_theta_0 = 0.2
        theta_0s, d_theta_0s = theta_0 + cloud_size * (np.random.rand(n_points) - 0.5), d_theta_0 + cloud_size * (
                np.random.rand(n_points) - 0.5)
        e_steps = 10

        # Generate Trajectories of different lengths and plot in phase space
        for A in [1.0, 3.0, 5.0]:
            for e_steps in [1000, 10000, 50000]:
                # Plot Evolution of different initial conditions
                fig = plt.figure(figsize=(8, 6))
                circle1 = plt.Circle((theta_0, d_theta_0), cloud_size, color='k', alpha=0.2)
                for idx, (theta_0i, d_theta_0i) in enumerate(zip(theta_0s, d_theta_0s)):
                    x = create_driven_data(tmax=T_MAX * 10, dt=dt, theta0=theta_0i, d_theta0=d_theta_0i, A=A, k=k, b=b)
                    i_x = np.where((np.abs(x[:, 0] - theta_0i) < 10e-3) & (np.abs(x[:, 1] - d_theta_0i) < 10e-3))[0]
                    if i_x.shape[0] == 0:
                        continue
                    elif i_x[0] + e_steps < x.shape[0]:
                        plt.plot(x[i_x[0]:i_x[0] + e_steps, 0], x[i_x[0]:i_x[0] + e_steps, 1], alpha=0.1)
                        xs = np.stack([x[i_x[0]], x[i_x[0] + e_steps]])
                        plt.scatter(xs[:, 0], xs[:, 1], s=8, c='k')
                # plt.legend()
                fig = plt.gcf()
                ax = fig.gca()
                ax.add_patch(circle1)
                plt.title(f'Nonlinear Evolution of Cloud of Initial Points for {e_steps} Steps')
                plt.xlabel(r'$\theta$')
                plt.ylabel(r'$\frac{d\theta}{dt}$')
                plt.savefig(f'{folder}/Figures/cloud_steps_driven{A}_{e_steps}.png')
                plt.close()

    # Create training and testing data
    xs = []
    num_test = 3
    theta_train = [0.1, 0.3, 0.5, 1.0, 1.2, 1.5, 2.0, 2.2, 2.5, 2.8]
    theta_test = [0.6, 1.3, 2.6]

    A_list_train = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    b_list_train = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # b_list_train = [0.5]

    A_list_test = [0.2, 0.5, 1.0, 1.5, 1.8]
    train_steps = int(train_h / dt)

    # Generate Train and Test Data
    for A in A_list_train:
        for b in b_list_train:
            x_train_full = []
            x_test_full = []
            print(f'A = {A}, b = {b}')
            fname = f'{folder}/Data/AltData_A{A}_b{b}.p'
            if not os.path.exists(f'{folder}/Data/AltData_A{A}_b{b}.p'):
                xs = []
                for theta_0i in theta_0s:
                    xi = create_driven_data(tmax=T_MAX, dt=dt, theta0=theta_0i, A=A, b=b, g=g, k=k)
                    n_batches = xi.shape[0] // (BATCH_SIZE*train_steps)
                    xs.append(xi[:BATCH_SIZE * train_steps * n_batches])
                    xi_base = create_data(tmax=T_MAX, dt=dt, theta0=theta_0i)
                x_train = xs
                x_train_full.append(np.vstack(x_train))
                # x_train = generate_datasets(fname, A, b, theta_train, dt, train_steps)
                # x_train_full.append(x_train)

                xs = []
                for theta_0i in theta_test:
                    xi = create_driven_data(tmax=T_MAX, dt=dt, theta0=theta_0i, A=A, b=b, g=g, k=k)
                    n_batches = xi.shape[0] // BATCH_SIZE
                    xs.append(xi[:BATCH_SIZE * train_steps * n_batches])
                    xi_base = create_data(tmax=T_MAX, dt=dt, theta0=theta_0i)
                x_test = xs
                x_test_full.append(np.vstack(x_test))

                x_train_full = np.vstack(x_train_full)
                x_test_full = np.vstack(x_test_full)
                torch.save((x_train_full, x_test_full), f'{folder}/Data/AltData_A{A}_b{b}.p')
            else:
                print('Loading Data')
                x_train_full, x_test_full = torch.load(f'{folder}/Data/AltData_A{A}_b{b}.p')

            # Train ODEnets
            if train_models:
                train_loader, test_loader = create_dataloader(x_train_full[::train_steps, :IN_DIM],
                                                              x_test_full[::train_steps, :IN_DIM],
                                                              batch_size=BATCH_SIZE)

                for integrator in ['euler', 'rk4']:
                    model_name = f'{folder}/Models/model_A{A}_b{b}_{integrator}.p'
                    hidden = 100
                    if os.path.isfile(model_name):
                        print(f'Loading {integrator} Model')
                        ODEnet, ode_loss_hist, best_loss = torch.load(model_name, map_location=device)
                        ODEnet, ode_loss_hist, best_loss = train(ODEnet, train_loader, method=integrator, dt=train_h,
                                                                 loss_list=ode_loss_hist, best_loss=best_loss, h=dt)
                        ODEnet = ODEnet.double().to(device)
                        torch.save((ODEnet, ode_loss_hist, best_loss), model_name)
                    else:
                        ODEnet = ShallowODE(in_dim=IN_DIM, hidden=hidden, out_dim=IN_DIM, Act=torch.nn.Tanh).double().to(device)
                        ODEnet, ode_loss_hist, best_loss = train(ODEnet, train_loader, method=integrator, dt=train_h, h=dt,
                                                                 fig_name=f'{folder}/loss_plot_{integrator}_alt.png')
                        torch.save((ODEnet, ode_loss_hist, best_loss), model_name)

            if test_traj:
                hs = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
                for h in hs:
                    print(f'h = {h}')
                    steps = int(h / dt)
                    x_test_traj = []
                    x_test_base_traj = []
                    xs = []
                    for theta_0i in theta_test:
                        xi = create_data(tmax=T_MAX, dt=h, theta0=theta_0i)
                        n_batches = xi.shape[0] // BATCH_SIZE
                        xs.append(xi[:BATCH_SIZE * n_batches])
                    x_test = xs
                    x_test_base_traj.append(np.vstack(x_test))
                    x_test_base_traj = np.vstack(x_test_base_traj)

                    xs_test = []
                    for theta_0i in theta_test:
                        xi = create_driven_data(tmax=T_MAX, dt=dt, theta0=theta_0i, A=A, b=b, g=g, k=k)
                        n_batches = xi.shape[0] // BATCH_SIZE
                        xs_test.append(xi[:BATCH_SIZE * n_batches])
                    x_test = np.vstack(xs_test)
                    x_test_traj.append(np.vstack(x_test))
                    x_test_traj = np.vstack(x_test_traj)

                    # diff = x_train_full - x_train_base_full
                    # if h == 0.1:
                    #     plt.figure(99)
                    #     plt.plot(diff)
                    #     plt.show()
                    #     plt.close()

                    sub_x_test = x_test_traj[::steps]
                    plt.figure(12)
                    plt.plot(x_test_traj[:, 2], x_test_traj[:, 0], label='RK4')
                    plt.plot(x_test_base_traj[:, 2], x_test_base_traj[:, 0], label='True')
                    plt.legend()
                    plt.savefig(f'{folder}/Full_traj{h}.png')
                    plt.close()
                    # train_sub_len, test_sub_len = sub_x_train.shape[0], sub_x_test.shape[0]
                    train_loader, test_loader = create_dataloader(sub_x_test,
                                                                  batch_size=BATCH_SIZE)
                    train_loader_base, test_loader_base = create_dataloader(x_test_base_traj,
                                                                            batch_size= BATCH_SIZE)
                    # Plot Trajectories
                    f = plt.figure(37)
                    for (inputs, targets), (base_inputs, base_targets) in zip(test_loader, test_loader_base):
                        # diff = (inputs[:, 0] - base_inputs[:, 0])
                        plt.plot(inputs[:, 2], inputs[:, 0])
                        plt.plot(base_inputs[:, 2], base_inputs[:, 0], 'k-')
                        # plt.plot(inputs[:, 2], diff)
                    plt.title(f'Trajectories for h={h}')
                    plt.savefig(f'{folder}/Figures/traj_h{h}.png')
                    plt.close()

            if check_losses:
                plt.figure(27)
                for integrator in integrators:
                    model_name = f'{folder}/Models/model_allA_{integrator}.p'
                    ODEnet, loss_list, _ = torch.load(model_name, map_location=device)
                    plt.plot(loss_list, label=f'{integrator}, driven')

                    model_name = f'{folder}/Models/model_{integrator}.p'
                    ODEnet, loss_list, best_state = torch.load(model_name, map_location=device)
                    plt.plot(loss_list, label=f'{integrator}, base')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                # plt.ylim([0, 0.004])
                plt.xlabel('Epoch')
                plt.legend()
                plt.savefig(f'{folder}/model_losses_A{A}.png')
                plt.close()

            if evaluate:
                with torch.no_grad():
                    models = {}
                    for model_suffix in [f'A{A}_b{b}_']:
                        # Evaluate the error of the models
                        errors = {}
                        hs = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
                        for integrator in integrators:
                            model_name = f'{folder}/Models/model_{model_suffix}{integrator}.p'
                            ODEnet, loss_list, _ = torch.load(model_name, map_location=device)
                            models[f'{integrator}{model_suffix}'] = ODEnet
                            for theta_0i in theta_test:
                                print(f'Integrator {integrator}, theta = {theta_0i}')
                                error = []
                                for h in hs:
                                    # f = plt.figure(52)
                                    steps = int(h/dt)
                                    target_list = []
                                    output_list = []
                                    diff_list = []
                                    x_test = create_driven_data(tmax=T_MAX, dt=dt, theta0=theta_0i, A=A, b=b, g=g, k=k)
                                    # x_test = create_data(tmax=T_MAX, dt=dt, theta0=theta_0i)
                                    n_batches = x_test.shape[0] // (BATCH_SIZE * train_steps)
                                    x_test = x_test[:n_batches*BATCH_SIZE*train_steps, :IN_DIM]
                                    train_loader, test_loader = create_dataloader(x_train_full[:1], x_test[::steps],
                                                                                  batch_size=BATCH_SIZE)
                                    for batch_idx, (inputs, targets) in enumerate(test_loader):
                                        outputs = ODEnet(inputs[:, :IN_DIM], h=h, dt=h, method=integrator)
                                        output_list.append(outputs.detach().cpu().numpy()[:, :OUT_DIM])
                                        target_list.append(targets[:, :OUT_DIM].cpu().numpy())
                                        # plt.plot(inputs[:, 2], inputs[:, 0])
                                    error.append(np.mean(np.linalg.norm(np.vstack(output_list) - np.vstack(target_list),
                                                                        axis=1) ** 2))
                                    # plt.title(f'Trajectories for h = {h}')
                                    # plt.show()
                                    # plt.close()
                                error = np.vstack(error)
                                errors[integrator] = error
                                # plt.figure(1)
                                print(f'Plotting Errors for {integrator}, model {model_suffix}')
                                # print(error.flatten())
                                plt.figure(1)
                                plt.plot(hs, error, 'o--',
                                         label=f'{integrator}, ' + r'$\theta_0$ =' + f'{theta_0i}')

                        plt.figure(1)
                        plt.yscale('log')
                        plt.xscale('log')
                        plt.legend(fontsize=14)
                        plt.tight_layout()
                        plt.savefig(f'{folder}/Figures/Errors/Models_errors_{model_suffix}vsH.png')
                        plt.close()
