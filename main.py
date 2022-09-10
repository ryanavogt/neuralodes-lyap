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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seed(seed=0):
    """Set one seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_simple_data(tmax=20, dt=1, theta0=2.0, d_theta0=0.0):
    """ Creates data for simple (linear) pendulum, for comparison purposes"""
    t = np.arange(0, tmax, dt)
    g = 9.81
    omega_0 = np.sqrt(g)
    S = np.sin(omega_0 * t)
    C = np.cos(omega_0 * t)
    d_theta_dt = -omega_0 * theta0 * S + d_theta0 * C
    theta = theta0 * C + d_theta0 / omega_0 * S
    return np.stack([theta, d_theta_dt], axis=1)


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
    return np.stack([theta, d_theta_dt], axis=1)


def create_dataloader(x, x_test=None, batch_size=BATCH_SIZE):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(np.asarray(x[0:-1]), dtype=torch.double).to(device),
        torch.tensor(np.asarray(x[1::]), dtype=torch.double).to(device),
    )

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    if x_test is None:
        x_test = x
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(np.asarray(x_test[0:-1]), dtype=torch.double).to(device),
        torch.tensor(np.asarray(x_test[1::]), dtype=torch.double).to(device),
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


# def train(ODEnet, train_loader, lr=LEARNING_RATE, wd=WEIGHT_DECAY, method='rk4', dt=0.1, fig_name=None):
#     optimizer_ODEnet = optim.Adam(ODEnet.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     best_loss = 100
#     criterion = torch.nn.MSELoss()
#
#     ode_loss_hist = []
#     print('ODENet Training')
#     epoch_losses = []
#     for epoch in range(1, NUMBER_EPOCHS):
#         epoch_loss = 0.0
#         for batch_idx, (inputs, targets) in enumerate(train_loader):
#             optimizer_ODEnet.zero_grad()
#             outputs = ODEnet(inputs.to(device), h=dt, dt=dt, method=method)
#             loss = criterion(outputs, targets.to(device))
#             loss.backward()
#             epoch_loss += loss.item()
#             optimizer_ODEnet.step()
#             ode_loss_hist.append(loss.item())
#         if epoch_loss < best_loss:
#             best_loss = epoch_loss
#             best_state = ODEnet.net.state_dict()
#
#         epoch_losses.append(epoch_loss / batch_idx)
#         if epoch % 50 == 0: print(f'Epoch: {epoch}, Loss: {epoch_loss / batch_idx}')
#         if fig_name is not None:
#             plt.figure(15)
#             plt.plot(range(epoch), epoch_losses)
#             plt.title(f'{integrator} Model Losses')
#             plt.xlabel('Epoch')
#             plt.ylabel('Loss')
#             plt.yscale('log')
#             plt.xlim([0, NUMBER_EPOCHS])
#             plt.savefig(fig_name)
#             plt.close()
#     ODEnet.net.load_state_dict(best_state)
#     return ODEnet, epoch_losses, best_loss

def train(ODEnet, train_loader, lr=LEARNING_RATE, wd=WEIGHT_DECAY, method='rk4', dt=0.0001, loss_list=[],
          h=0.1, best_loss=100, fig_name=None):
    optimizer_ODEnet = optim.Adam(ODEnet.parameters(), lr=lr, weight_decay=wd)
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
            loss = criterion(outputs, targets)
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


if __name__ == '__main__':
    dt = 0.1
    alt_dt = 0.0001
    eps = 0.01
    theta_0 = 0.5
    d_theta_0 = 0
    omega_0 = np.sqrt(9.81)
    N_points = 500
    LE_points = 1000
    T_MAX = N_points * dt
    n_points = 100
    cloud_size = 0.4
    set_seed(5544)
    folder = 'Standard'
    plot = False

    theta_0s, d_theta_0s = theta_0 + cloud_size * (np.random.rand(n_points) - 0.5), d_theta_0 + cloud_size * (
            np.random.rand(n_points) - 0.5)
    e_steps = 10

    # Generate Trajectories of different lengths and plot in phase space
    if plot:
        for e_steps in [10, 25, 60, 100, 500]:
            # Plot Evolution of different initial conditions
            fig = plt.figure(figsize=(8, 6))
            circle1 = plt.Circle((theta_0, d_theta_0), cloud_size, color='k', alpha=0.2)
            for idx, (theta_0i, d_theta_0i) in enumerate(zip(theta_0s, d_theta_0s)):
                x = create_data(tmax=T_MAX * 10, dt=dt, theta0=theta_0i, d_theta0=d_theta_0i)
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
            plt.savefig(f'{folder}/Figures/cloud_steps{e_steps}.png')
            plt.close()

            # Plot Linear Pendulum to show it is not chaotic
            fig = plt.figure(figsize=(8, 6))
            circle1 = plt.Circle((theta_0, d_theta_0), cloud_size, color='k', alpha=0.2)
            for idx, (theta_0i, d_theta_0i) in enumerate(zip(theta_0s, d_theta_0s)):
                x = create_simple_data(tmax=T_MAX * 10, dt=dt, theta0=theta_0i, d_theta0=d_theta_0i)
                i_x = np.where((np.abs(x[:, 0] - theta_0i) < 10e-3) & (np.abs(x[:, 1] - d_theta_0i) < 10e-3))[0]
                if i_x.shape[0] == 0:
                    continue
                elif i_x[0] + e_steps < x.shape[0]:
                    plt.plot(x[:e_steps, 0], x[i_x[0]:i_x[0] + e_steps, 1], alpha=0.1)
                    xs = np.stack([x[i_x[0]], x[i_x[0] + e_steps]])
                    plt.scatter(xs[:, 0], xs[:, 1], s=8, c='k')
            # plt.legend()
            fig = plt.gcf()
            ax = fig.gca()
            ax.add_patch(circle1)
            plt.title(f'Simple (Linear) Evolution of Cloud of Initial Points for {e_steps} Steps')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$\frac{d\theta}{dt}$')
            plt.savefig(f'{folder}/Figures/cloud_steps{e_steps}_simple.png')
            plt.close()

    # Create training and testing data
    theta_0_train = [0.1, 0.3, 0.5, 1.0, 1.2, 1.5, 2.0, 2.2, 2.5, 2.8]
    theta_0_test = [0.6, 1.3, 2.6]

    xs = []
    x_train_full = []
    for theta_0i in theta_0_train:
        xi = create_data(tmax=T_MAX, dt=dt, theta0=theta_0i)
        n_batches = xi.shape[0] // BATCH_SIZE
        xs.append(xi[:BATCH_SIZE * n_batches])
    x_train = xs
    x_train_full.append(np.vstack(x_train))
    x_train_full = np.vstack(x_train_full)

    xs = []
    x_test_full = []
    for theta_0i in theta_0_test:
        xi = create_data(tmax=T_MAX, dt=dt, theta0=theta_0i)
        n_batches = xi.shape[0] // BATCH_SIZE
        xs.append(xi[:BATCH_SIZE])
    x_test = xs
    x_test_full.append(np.vstack(x_test))
    x_test_full = np.vstack(x_test_full)

    # for theta_0i in theta_0_train:
    #     xi = create_data(tmax=T_MAX, dt=dt, theta0=theta_0i)
    #     n_batches = xi.shape[0] // BATCH_SIZE
    #     xs.append(xi[:BATCH_SIZE * n_batches])
    # x_train = np.vstack(xs)
    # xs_test = []
    # for theta_0i in theta_0_test:
    #     xi = create_data(tmax=T_MAX, dt=dt, theta0=theta_0i)
    #     n_batches = xi.shape[0] // BATCH_SIZE
    #     xs_test.append(xi[:BATCH_SIZE * n_batches])
    # x_test = np.vstack(xs_test)
    # torch.save((x_train, x_test), f'{folder}/base_data.p')
    # print('Data Saved')

    alt_steps = int(dt/alt_dt)
    train_alt, test_alt = torch.load(f'{folder}/AltData.p')
    # train_loader, test_loader = create_dataloader(x_train_full, x_test_full)
    train_loader, test_loader = create_dataloader(train_alt[::alt_steps, :2], test_alt[::alt_steps, :2])
    errors = {}

    f = plt.figure()
    for integrator in ['euler', 'rk4']:
        hidden = 100
        model_name = f'{folder}/Models/model_{integrator}.p'

        if (integrator != 'dy_sys') and (integrator != 'simple'):
            if os.path.isfile(model_name):
                print(f'Loading {integrator} Model')
                ODEnet, ode_loss_hist, best_state = torch.load(model_name, map_location=device)
                ODEnet = ODEnet.double()
            else:
                ODEnet = ShallowODE(in_dim=2, hidden=hidden, out_dim=2, Act=torch.nn.Tanh).double().to(device)
                ODEnet, ode_loss_hist, best_state = train(ODEnet, train_loader, method=integrator, dt=dt,
                                                          h=dt, fig_name=f'{folder}/loss_plot_{integrator}_base.png')
                torch.save((ODEnet, ode_loss_hist, best_state), model_name)

        # Evaluate the model and calculate LEs --
        # hs = [0.05, 0.1, 0.5, 1, 5]
        hs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        f = plt.figure(1)
        LE_theta = []
        # theta_0_list = [0.0, 0.2, 0.5, 1.0, 2.0]
        theta_0_list = [0.6, 1.3, 2.6]
        alt = False
        if alt:
            suffix = '_leap'
        else:
            suffix = ''
        for theta_0 in theta_0_list:
            print(r'$\theta_0$ = ' + f'{theta_0}')
            error = []
            LE_list = []
            for h in hs:
                x = torch.DoubleTensor(create_data(tmax=T_MAX, dt=h, theta0=theta_0))
                _, test_loader = create_dataloader(x)
                le_file = f'{folder}/LEs/{integrator}_h{h}_eps{eps}_theta{theta_0}_LEs{suffix}.p'
                if integrator in ['dy_sys', 'simple']:
                    if os.path.isfile(le_file):
                        LEs, rvals, J = torch.load(le_file, map_location=device)
                    else:
                        x_le = torch.DoubleTensor(create_data(tmax=np.min((LE_points * h, LE_points)), dt=h,
                                                              theta0=theta_0, d_theta0=d_theta_0)).to(device)
                        LEs, rvals, J = dysys_lyapunov(x_le, h=h, system=integrator, eps=eps, alt=False)
                        torch.save((LEs, rvals, J), le_file)
                    plt.figure(3)
                    plt.close()
                else:
                    target_list = []
                    output_list = []
                    diff_list = []
                    for batch_idx, (inputs, targets) in enumerate(test_loader):
                        outputs = ODEnet(inputs, h=h, dt=h, method=integrator)
                        output_list.append(outputs.detach().cpu().numpy())
                        target_list.append(targets.cpu().numpy())

                    error.append(np.mean(np.linalg.norm(np.vstack(output_list) - np.vstack(target_list), axis=1) ** 2))

                    if os.path.isfile(le_file):
                        LEs, rvals, J = torch.load(le_file, map_location=device)
                    else:
                        x_le = create_data(tmax=np.min((LE_points * h, LE_points)), dt=h, theta0=theta_0)
                        _, le_loader = create_dataloader(x_le, batch_size=x_le.shape[0])
                        le_inputs, _ = next(iter(le_loader))
                        LEs, rvals, J = ode_lyapunov(ODEnet.net, le_inputs, h=h, integrator=integrator)
                        torch.save((LEs, rvals, J), le_file)

                if type(J) == torch.Tensor:
                    J = J.cpu()
                plt.figure(3)
                plt.plot(J[:200, 0, 0], label='J(0,0)')
                plt.plot(J[:200, 0, 1], label='J(0,1)')
                plt.plot(J[:200, 1, 0], label='J(1,0)')
                plt.plot(J[:200, 1, 1], label='J(1,1)')
                plt.legend()
                plt.xlabel('Time Step')
                plt.title(f'Jacobian values for {integrator}, h = {h}, ' + r'$\theta_0$ = ' f'{theta_0}')
                plt.ylim([-10, 10])
                plt.savefig(f'{folder}/Figures/Jacs_{integrator}_h{h}_theta{theta_0}{suffix}.png')
                plt.close()

                LE_list.append(LEs)
                print(f'LEs for {integrator}, h = {h}:   \t {LEs}')
                f_rval = plt.figure()
                plt.title(f'LE value evolution over time, h={h},' + r' $\theta_0$=' + f'{theta_0}')
                plt.xlabel('Time Step')
                plt.ylabel('LE value')
                if type(rvals) == torch.Tensor:
                    rvals = rvals.cpu()
                    rv_plot = torch.log2(rvals.diagonal(dim1=-2, dim2=-1)).cumsum(dim=0).detach().cpu()
                    denom = torch.ones_like(rv_plot)
                else:
                    rv_plot = np.log2(rvals.diagonal(axis1=-2, axis2=-1)).cumsum(axis=0)
                    denom = np.ones_like(rv_plot).cumsum(axis=0)
                plt.plot(rv_plot / denom)
                plt.legend([r'$\lambda_1$', r'$\lambda_2$'])
                plt.ylim([LEs.mean().cpu() - 0.5, LEs.mean().cpu() + 0.5])
                plt.savefig(f'{folder}/Figures/rvals_{integrator}_h{h}_theta{theta_0}{suffix}.png')
                plt.close(f_rval.number)

                if h == 0.1:
                    LE_theta.append(LEs)

            if integrator in ['euler', 'rk4']:
                error = np.vstack(error)
                print(error.flatten())
                errors[integrator] = error
                plt.figure(1)
                plt.plot(hs, error, 'o--', label=f'{integrator} ' + r'$\theta_0$=' + f'{theta_0}')
            else:
                plt.figure(3)
                plt.legend(title='h')
                plt.xlabel('t')
                plt.tight_layout()
                plt.title('Generated Data as function of h')
                plt.savefig(f'{folder}/Figures/traj_{integrator}{suffix}.png', dpi=300)
                plt.close()
            if type(LE_list[0]) == torch.Tensor:
                LE_list = torch.vstack(LE_list).cpu().numpy()
            else:
                LE_list = np.vstack(LE_list)
            plt.figure(2)
            plt.scatter(np.expand_dims(np.array(hs), axis=1).repeat(2, axis=-1), LE_list, label=integrator)
            plt.scatter(np.expand_dims(np.array(hs), axis=1), LE_list.sum(axis=-1), c='k', marker='+', alpha=0.5)
        if type(LEs) == torch.Tensor:
            LE_theta = torch.vstack(LE_theta).cpu().numpy()
        else:
            LE_theta = np.vstack(LE_theta)
        plt.figure(4)
        plt.scatter(torch.Tensor(theta_0_list[1:]).unsqueeze(-1).repeat(1, 2), LE_theta[1:])
        plt.xlabel(r'$\theta_0$')
        plt.ylabel('Lyapunov Exponent')
        plt.title(r'LE vs $\theta_0$ for ' + f'{integrator}')
        plt.savefig(f'{folder}/Figures/LE_v_theta_{integrator}{suffix}.png')
        plt.close()

    plt.figure(2)
    plt.legend(title=r'method')
    plt.xscale('log')
    plt.xlabel('h')
    plt.ylabel('Lyapunov Exponent')
    plt.savefig(f'{folder}/Figures/LE_v_h{suffix}.png')
    plt.close()

    print('Error Plot')
    plt.figure(1)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{folder}/Models_errors_v_h.png')
