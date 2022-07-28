import numpy as np
import scipy as sci

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from scipy.special import ellipj, ellipk
from ODE_Jacs import *
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


def create_simple_data(tmax=20, dt=1, theta0=2.0, d_theta0 = 0.0):
    t = np.arange(0, tmax, dt)
    g = 9.81
    omega_0 = np.sqrt(g)
    S = np.sin(omega_0*t)
    C = np.cos(omega_0*t)
    d_theta_dt = -omega_0*theta0*S + d_theta0*C
    theta = theta0*C + d_theta0/omega_0*S
    return np.stack([theta, d_theta_dt], axis=1)


def create_data(tmax=20, dt=1, theta0=2.0, d_theta0 = 0.0):
    """Solution for the nonlinear pendulum in theta space."""
    t = np.arange(0, tmax, dt)
    g = 9.81
    alpha = np.arccos(np.cos(theta0) - 0.5*d_theta0**2/g)
    S = np.sin(0.5*alpha)
    K_S = ellipk(S**2)
    omega_0 = np.sqrt(g)
    sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
    theta = 2.0*np.arcsin( S*sn )
    d_sn_du = cn*dn
    d_sn_dt = -omega_0 * d_sn_du
    d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
    return np.stack([theta, d_theta_dt], axis=1)


def create_dataloader(x, batch_size=BATCH_SIZE):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(np.asarray(x[0:-1]), dtype=torch.double),
        torch.tensor(np.asarray(x[1::]), dtype=torch.double),
    )

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(np.asarray(x[0:-1]), dtype=torch.double),
        torch.tensor(np.asarray(x[1::]), dtype=torch.double),
    )

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def euler_step_func(f, x, dt, monitor = False):
    """The 'forward' Euler, a one stage Runge Kutta."""
    k1 = f(x)
    x_out = x + dt * k1
    if monitor:
        return x_out, k1
    else:
        return x_out


def rk4_step_func(f, x, dt, monitor = False):
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
    x1 = x + 1.0/3.0 * dt * k1
    k2 = f(x1)
    x2 = x + (-1.0/3.0 * k1 + k2) * dt
    k3 = f(x2)
    x3 = x + (k1 - k2 + k3)*dt
    k4 = f(x3)
    x_out = x + dt * 1.0/8.0 * (k1 + 3 * k2 + 3 *k3 + k4)
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
        if method=='euler':
            #print(method)
            #for i in range(int(dt/h)):
            x = euler_step_func(self.net, x, dt)
            return x
        elif method=='rk4':
            #print(method)
            #for i in range(int(dt/h)):
            x = rk4_step_func(self.net, x, dt)
            return x
        elif method=='rk4_38':
            #print(method)
            #for i in range(int(dt/h)):
            x = rk4_38_step_func(self.net, x, dt)
            return x


def train(ODEnet, train_loader, lr=LEARNING_RATE, wd=WEIGHT_DECAY, method='rk4', dt=0.1):

    optimizer_ODEnet = optim.Adam(ODEnet.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    criterion = torch.nn.MSELoss()
    ode_loss_hist = []
    print('ODENet Training')
    for epoch in range(1, NUMBER_EPOCHS):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer_ODEnet.zero_grad()
            outputs = ODEnet(inputs, h=dt, dt=dt, method=method)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_ODEnet.step()
            ode_loss_hist.append(loss.item())

        if epoch % 50 == 0: print(f'Epoch: {epoch}, Loss: {loss.item()}')

    return ODEnet, ode_loss_hist


if __name__ == '__main__':
    dt = 0.1
    eps = -1
    theta_0 = 2.0
    d_theta_0 = 2.0
    omega_0 = np.sqrt(9.81)
    N_points = 500
    LE_points = 10000
    T_MAX = N_points * dt
    n_points = 100
    cloud_size = 0.8
    set_seed(5544)

    theta_0s, d_theta_0s = theta_0+cloud_size*(np.random.rand(n_points)-0.5), d_theta_0+cloud_size*(np.random.rand(n_points)-0.5)
    e_steps = 10

    for e_steps in [10, 25, 60, 100, 500]:
        fig = plt.figure(figsize=(8, 6))
        circle1 = plt.Circle((theta_0, d_theta_0), cloud_size, color='k', alpha=0.2)
        for idx, (theta_0i, d_theta_0i) in enumerate(zip(theta_0s, d_theta_0s)):
            x = create_data(tmax=T_MAX*10, dt=dt, theta0=theta_0i, d_theta0=d_theta_0i)
            i_x = np.where((np.abs(x[:, 0] - theta_0i)<10e-3) & (np.abs(x[:, 1]-d_theta_0i)<10e-3))[0]
            if i_x.shape[0] == 0:
                continue
            elif i_x[0]+e_steps < x.shape[0]:
                plt.plot(x[i_x[0]:i_x[0] + e_steps, 0], x[i_x[0]:i_x[0] + e_steps, 1], alpha=0.1)
                xs = np.stack([x[i_x[0]], x[i_x[0]+e_steps]])
                plt.scatter(xs[:, 0], xs[:, 1], s=8, c = 'k')
        # plt.legend()
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_patch(circle1)
        plt.title(f'Nonlinear Evolution of Cloud of Initial Points for {e_steps} Steps')
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\frac{d\theta}{dt}$')
        plt.savefig(f'Figures/cloud_steps{e_steps}.png')
        plt.close()

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
        plt.savefig(f'Figures/cloud_steps{e_steps}_simple.png')
        plt.close()

    train_loader, test_loader = create_dataloader(x)
    errors = {}

    y_in, _ = next(iter(test_loader))
    nonlinear_pendulum(y_in)

    f = plt.figure()
    for integrator in ['euler', 'rk4', 'dy_sys']:
    # for integrator in ['dy_sys']:
        hidden = 100
        model_name = f'Models/model_{integrator}.p'

        if (integrator != 'dy_sys') and (integrator != 'simple'):
            if os.path.isfile(model_name):
                print(f'Loading {integrator} Model')
                ODEnet, ode_loss_hist = torch.load(model_name, map_location=device)
            else:
                ODEnet = ShallowODE(in_dim=2, hidden=hidden, out_dim=2, Act=torch.nn.Tanh).double()
                ODEnet, ode_loss_hist = train(ODEnet, train_loader, method=integrator, dt=dt)
                torch.save((ODEnet, ode_loss_hist), model_name)
        error = []
        LE_list = []

        # Evaluate the model --
        hs = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 10]
        # hs = [0.001, 0.01, 0.1,  1, 10]
        f = plt.figure(1)
        for h in hs:
            x = torch.Tensor(create_data(tmax=T_MAX, dt=h)).double()
            _, test_loader = create_dataloader(x)
            le_file = f'LEs/{integrator}_h{h}_LEs.p'
            if integrator == 'dy_sys':
                if os.path.isfile(le_file):
                    LEs, rvals, J = torch.load(le_file, map_location = device)
                    print(f'LEs for {integrator}, h = {h}: {LEs.data}')
                else:
                    x_le = torch.Tensor(create_data(tmax= np.min((LE_points*h, LE_points)), dt=h)).double()
                    LEs, rvals, J = dysys_lyapunov(x_le, h=h, system='pendulum')
                    torch.save((LEs, rvals, J), le_file)
                plt.figure(3)
                # expected_J = np.abs(omega_0**2*np.sin(x_le[:, 0]))
                # plt.plot(torch.ones_like(x_le[:, 0]).cumsum(dim=0) * h - h, rvals[:, 0, 0].cpu(), label='rvals')
                # plt.plot(torch.ones_like(x_le[:, 0]).cumsum(dim=0) * h - h, expected_J[:], label='exp')
                # plt.plot(torch.ones_like(x_le[:, 0]).cumsum(dim=0) * h - h, np.abs(x_le[:, 1]), label='exp')
                # plt.legend()
                # plt.show()
                plt.close()
            else:
                target_list = []
                output_list = []
                diff_list = []
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    outputs = ODEnet(inputs, h=h, dt=h, method=integrator)
                    output_list.append(outputs.detach().numpy())
                    target_list.append(targets.numpy())

                error.append(np.mean(np.linalg.norm(np.vstack(output_list)-np.vstack(target_list), axis=1)**2))

                if os.path.isfile(le_file):
                    LEs, rvals, J = torch.load(le_file, map_location=device)
                    print(f'LEs for {integrator}, h = {h}: {LEs.data}')
                else:
                    x_le = create_data(tmax= np.min((LE_points*h, LE_points)), dt=h)
                    _, le_loader = create_dataloader(x_le, batch_size=x_le.shape[0])
                    le_inputs, _ = next(iter(le_loader))
                    LEs, rvals, J = ode_lyapunov(ODEnet.net, le_inputs, h=h, integrator=integrator)
                    torch.save((LEs, rvals, J), le_file)

            plt.figure(3)
            plt.plot(J[:200, 0, 0], label='J(0,0)')
            plt.plot(J[:200, 0, 1], label='J(0,1)')
            plt.plot(J[:200, 1, 0], label='J(1,0)')
            plt.plot(J[:200, 1, 1], label='J(1,1)')
            plt.legend()
            plt.xlabel('Time Step')
            plt.title(f'Jacobian values for {integrator}, h = {h}')
            plt.savefig(f'Figures/Jacs_{integrator}_h{h}.png')
            plt.close()

            LE_list.append(LEs)
            print(f'LEs for {integrator}, h = {h}:   \t {LEs}')
            f_rval = plt.figure()
            plt.title(f'LE value evolution over time, eps = {eps}')
            plt.xlabel('Time Step')
            plt.ylabel('LE value')
            plt.legend([r'$\lambda_1$', r'$\lambda_2$'])
            rv_plot = torch.log2(rvals.diagonal(dim1=-2, dim2=-1)).cumsum(dim=0).detach().cpu()
            plt.plot(rv_plot / (torch.ones_like(rv_plot)).cumsum(dim=0))
            plt.ylim([-0.5, 2.5])
            plt.savefig(f'Figures/rvals_{integrator}_h{h}.png')
            plt.close(f_rval.number)

        if integrator in ['euler', 'rk4']:
            error = np.vstack(error)
            errors[integrator] = error
            plt.figure(1)
            plt.plot(hs, error, 'o--', label=integrator)
        else:
            plt.figure(3)
            plt.legend(title='h')
            plt.xlabel('t')
            plt.tight_layout()
            plt.title('Generated Data as function of h')
            plt.savefig(f'Figures/traj_{integrator}.png', dpi=300)
            plt.close()
        LE_list = torch.vstack(LE_list).cpu()
        plt.figure(2)
        plt.scatter(torch.Tensor(hs).unsqueeze(1).repeat(1, 2), LE_list, label=integrator)

plt.figure(2)
plt.legend(title = 'Method')
plt.xscale('log')
plt.xlabel('h')
plt.ylabel('Lyapunov Exponent')
plt.savefig(f'Figures/LE_v_h.png')
plt.close()

plt.figure(1)
plt.yscale('log')
plt.xscale('log')
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig(f'Models_errors_v_h.png')
plt.show()

