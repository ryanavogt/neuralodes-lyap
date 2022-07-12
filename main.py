# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import scipy as sci

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

import torch
import torchdiffeq
import torch.optim as optim
from ODE_Jacs import *

BATCH_SIZE = 16
WEIGHT_DECAY = 0
LEARNING_RATE = 1e-2
NUMBER_EPOCHS = 60
LE_BATCH_SIZE = 500
T_MAX = 20
METHOD = 'rk4'


def set_seed(seed=0):
    """Set one seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_data(b=0.0, k=0.0, tmax=20, dt=0.4):
    grid = np.arange(0, tmax, dt)

    # Do the numerical integration of the equations of motion
    x = solve_ivp(pendulum, (0, tmax), y0=np.array([2.6, 0.0]), args=(b, k), t_eval=grid)
    x2 = solve_ivp(pendulum, (0, tmax), y0=np.array([1.4, 0.0]), args=(b, k), t_eval=grid)

    return x, x2, grid


def plot_pendulum(b=0.0, k=0.0):
    theata, velocity = np.meshgrid(np.arange(-2 * np.pi, 2 * np.pi, 0.01), np.arange(-2.5, 2.5, 0.01))
    theata_dot = velocity
    velocity_dot = b * velocity - np.sin(theata) + k

    plt.figure(figsize=(15, 5))
    color = np.hypot(theata_dot, velocity_dot)
    plt.streamplot(theata, velocity, theata_dot, velocity_dot, color=color, linewidth=1, cmap=plt.cm.RdBu,
                   density=2, arrowstyle='->', arrowsize=1.5)

    plt.tight_layout()


def pendulum(t, x, b=0, k=0, drag=0.0):
    theta, velocity = x
    theata_dot = velocity
    velocity_dot = -b * velocity - np.sin(theta) + drag * np.cos(k * t)

    return theata_dot, velocity_dot


def plot_data(x, x2, figsize=(15, 5)):
    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(121)
    ax.plot(grid, x.y[0], 'o--', label=r'$\theta(0)=2.4$', c='#dd1c77')
    ax.plot(grid, x2.y[0], 's--', label=r'$\theta(0)=2.0$', c='#2c7fb8')
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_ylabel(r'angular displacement, ${\theta}(t)$', fontsize=14)
    ax.set_xlabel('time, t', fontsize=14)
    ax.legend(loc='lower right', fontsize=14)

    ax2 = fig.add_subplot(122)
    ax2.plot(grid, x.y[1], 'o--', label=r'$\theta(0)=2.4$', c='#dd1c77')
    ax2.plot(grid, x2.y[1], 's--', label=r'$\theta(0)=2.0$', c='#2c7fb8')
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='both', which='minor', labelsize=14)
    ax2.set_ylabel(r'angular velocity, ${v}(t)$', fontsize=14)
    ax2.set_xlabel('time, t', fontsize=14)
    ax2.legend(loc='lower right', fontsize=14)
    plt.show()


def create_dataloader(x, x2, batch_size=BATCH_SIZE):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(np.asarray([x.y[0][0:-1], x.y[1][0:-1]]).T, dtype=torch.float),
        torch.tensor(np.asarray([x.y[0][1:], x.y[1][1:]]).T, dtype=torch.float),
        torch.tensor(np.asarray([x.t[1:] - x.t[0:-1]]).reshape(-1, 1), dtype=torch.float),
    )

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(np.asarray([x2.y[0][0:-1], x2.y[1][0:-1]]).T, dtype=torch.float),
        torch.tensor(np.asarray([x2.y[0][1:], x2.y[1][1:]]).T, dtype=torch.float),
        torch.tensor(np.asarray([x2.t[1:] - x2.t[0:-1]]).reshape(-1, 1), dtype=torch.float),
    )

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    le_loader = torch.utils.data.DataLoader(dataset, batch_size=LE_BATCH_SIZE, shuffle=False)
    return train_loader, test_loader, le_loader


def create_predset(b= 0.0, k= 0.0, tmax= 20, h= 0.4, length= BATCH_SIZE):
    _, x2, grid = create_data(b, k, tmax, h)

    y0 = torch.tensor(np.asarray([x2.y[0][0:-1], x2.y[1][0:-1]]).T, dtype=torch.float)
    y1 = torch.tensor(np.asarray([x2.y[0][1:], x2.y[1][1:]]).T, dtype=torch.float)
    t = torch.tensor(np.asarray([x2.t[1:] - x2.t[0:-1]]).reshape(-1, 1), dtype=torch.float)
    return y0, y1, t


def shallow(in_dim, hidden, out_dim, Act=torch.nn.Tanh):
    """Just make a shallow network. This is more of a macro."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        Act(),
        torch.nn.Linear(hidden, out_dim), )


class ShallowNet(torch.nn.Module):
    """Just a basic shallow network"""

    def __init__(self, in_dim, out_dim, hidden=10, Act=torch.nn.Tanh):
        super(ShallowNet, self).__init__()
        self.net = shallow(in_dim, hidden, out_dim, Act=Act)

    def forward(self, x):
        return self.net(x)


class ShallowSkipNet(torch.nn.Module):
    """A basic shallow network with a skip connection"""

    def __init__(self, in_dim, out_dim, hidden=10, Act=torch.nn.Tanh, eps=1.0):
        super(ShallowSkipNet, self).__init__()
        self.eps = eps
        self.net = shallow(in_dim, hidden, out_dim, Act=Act)

    def forward(self, x):
        return x + self.eps * self.net(x)


class ShallowODE(torch.nn.Module):
    """A basic shallow network that takes in a t as well"""

    def __init__(self, in_dim, out_dim, hidden=10, Act=torch.nn.Tanh):
        super(ShallowODE, self).__init__()
        self.net = shallow(in_dim, hidden, out_dim, Act=Act)

    def forward(self, t, x):
        return self.net(x)


def train(net, ODEnet, train_loader, lr=LEARNING_RATE, wd=WEIGHT_DECAY, method='rk4'):
    optimizer_net = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer_ODEnet = optim.Adam(ODEnet.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    criterion = torch.nn.MSELoss()
    loss_hist = []
    print('Standard Net Training')
    for epoch in range(1, NUMBER_EPOCHS):
        for batch_idx, (inputs, targets, dt) in enumerate(train_loader):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_net.step()
            optimizer_net.zero_grad()
        loss_hist.append(loss.item())

        if epoch % 10 == 0: print(f'Epoch: {epoch}, Loss: {loss.item()}')

    ode_loss_hist = []
    print('ODENet Training')
    for epoch in range(1, NUMBER_EPOCHS):
        for batch_idx, (inputs, targets, dt) in enumerate(train_loader):
            outputs = torchdiffeq.odeint(ODEnet, inputs, torch.tensor([0, dt[0]]).float(), method=method)[-1, :, :]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_ODEnet.step()
            optimizer_ODEnet.zero_grad()
        ode_loss_hist.append(loss.item())

        if epoch % 10 == 0: print(f'Epoch: {epoch}, Loss: {loss.item()}')

    return net, ODEnet, loss_hist, ode_loss_hist


def evaluate(net, ODEnet, test_loader, method='rk4', h= 0.1):
    preds_net = []
    for batch_idx, (inputs, targets, dt) in enumerate(test_loader):
        preds = net(inputs).detach().numpy()
        preds_net.append(preds)
    preds_net = np.vstack(preds_net)

    preds_ODEnet = []
    for batch_idx, (inputs, targets, dt) in enumerate(test_loader):
        preds = torchdiffeq.odeint(ODEnet, inputs, torch.tensor([0, h]).float(), method=method)[-1, :,
                :].detach().numpy()
        preds_ODEnet.append(preds)
    preds_ODEnet = np.vstack(preds_ODEnet)

    return preds_net, preds_ODEnet


def predict(ODEnet, test_loader, method = 'rk4', h= 0.1):
    preds_odenet = []
    inputs, targets, dt = next(iter(test_loader))
    # print(f't steps: {T_MAX/h}')
    for batch_idx in range(int(T_MAX/h)):
        if batch_idx == 0:
            preds = torchdiffeq.odeint(ODEnet, inputs[:1], torch.tensor([0, h]).float(), method=method)[-1, :,
                :].detach().numpy()
            preds_odenet.append(preds)
        else:
            preds = torchdiffeq.odeint(ODEnet, torch.Tensor(preds), torch.tensor([0, h]).float(), method=method)[-1, :,
                    :].detach().numpy()
            preds_odenet.append(preds)
    preds_odenet = np.vstack(preds_odenet)
    return preds_odenet


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dt = 0.1
    eps = 0.1
    x, x2, grid = create_data(tmax=T_MAX, dt=dt)
    # plot_pendulum()
    train_loader, test_loader, le_loader = create_dataloader(x, x2)
    integrator = 'euler'
    hidden = 100

    set_seed(10 ** 3)
    net = ShallowNet(in_dim=2, hidden=hidden, out_dim=2, Act=torch.nn.Tanh)

    set_seed(10 ** 3)
    ODEnet = ShallowODE(in_dim=2, hidden=hidden, out_dim=2, Act=torch.nn.Tanh)

    net, ODEnet, loss_hist, ode_loss_hist = train(net, ODEnet, train_loader)
    batch_y, batch_y2, batch_t = next(iter(le_loader))
    k1_test = ODEnet(batch_t, batch_y)
    print(f'batch_t shape: {batch_t.shape}')
    preds_list = []
    # pred_f = plt.figure()
    # plt.plot(batch_t[:, 0].cumsum(dim=0), batch_y2[:, 0], label= 'actual')
    error = []

    hs = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
    for h in hs:
        preds_net, preds_odenet = evaluate(net, ODEnet, test_loader, method=integrator, h= h)
        # y0, y1, t = create_predset(tmax= T_MAX, h= h)
        # plt.plot(t.cumsum(dim=0), y1[:, 0], label= f'actual, h={h}')
        # full_preds = predict(ODEnet, test_loader, method=integrator, h= h)
        # plt.plot(torch.arange(start= 0, end = T_MAX, step = h), full_preds[:, 0], label=f'pred: h={h}')
        error.append(torch.linalg.norm(torch.Tensor(preds_odenet[-1])-batch_y2[-1]))
        preds_list.append(preds_odenet)
        LEs, rvals, ynew = ode_lyapunov(ODEnet, batch_t, batch_y, h=h, integrator=integrator)
        # print(rvals.diagonal(dim1=-2, dim2=-1).shape)
        print(f'LEs for {integrator}, h = {h}:   \t {LEs}')
    error = torch.vstack(error)
    plt.figure()
    plt.title(f'Errors for {integrator}, ' + r'$\Delta t =$' + f'{dt}')
    plt.scatter(hs, error)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('h')
    plt.ylabel('error')
    plt.savefig(f'Errors_{integrator}.png')

    f = plt.figure()
    plt.title(f'LE value evolution over time, eps = {eps}')
    plt.xlabel('Time Step')
    plt.ylabel('LE value')
    plt.legend([r'$\lambda_1$', r'$\lambda_2$'])
    plt.plot(torch.log2(rvals.diagonal(dim1=-2, dim2=-1)).cumsum(dim=0).detach() / ((batch_t / dt).cumsum(dim=0)))
    #

    # plt.plot(ynew[:], label='predicted')
    # plt.legend()
    # plt.savefig(f'y_{integrator}.png')
    plt.show()