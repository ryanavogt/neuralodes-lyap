import numpy as np
import torch


def net_der(model, xi, yi):
    W = model.net[0].weight
    b = model.net[0].bias
    Wout = model.net[-1].weight
    bout = model.net[-1].bias
    arg = yi @ W.T + b
    der = (1 - torch.tanh(arg) ** 2) * W.T
    # print(der.shape)
    der_out = der @ Wout.T
    # print(f'der out shape: {der_out.shape}')
    return der_out


def rk_alt(model, xi, yi, h=0.01):
    k1 = model(0, yi)
    k2 = model(0, yi + h * k1 * 1./3)
    k3 = model(0, yi + h * (k2 - k1 * 1./3))
    k4 = model(0, yi + h * (k1 - k2 + k3))
    ynew = (k1 + 3 * (k2 + k3) + k4) * h * 0.125
    return ynew


def ode_jacobian(model, xi, yi, h=0.01, integrator='rk4'):
    if integrator == 'rk4':
        order = 4
        A = torch.Tensor([[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]])
        b = torch.Tensor([1 / 8, 3 / 8, 3 / 8, 1 / 8])
        c = torch.Tensor([0, 1 / 3, 2 / 3, 1])

    if integrator == 'euler':
        order = 1
        A=torch.Tensor([[0]])
        b=torch.Tensor([1])
        c=torch.Tensor([0])

    if integrator == 'midpoint':
        order = 2
        A = torch.Tensor([[0, 0], [1/2, 0]])
        b = torch.Tensor([0, 1])
        c = torch.Tensor([0, 1/2])

    # Calculate quantities for integrator
    K = torch.zeros((order + 1, yi.shape[-1]))
    K[0] = yi
    dy = torch.zeros((order, yi.shape[-1]))
    # Calculate K's of integration method
    for i in range(order):
        dy[i] = (A[i].unsqueeze(0) @ K[1:])*h
        k = model(xi + c[i] * h, yi + dy[i])
        K[i + 1] = k
    ynew = yi + (b.unsqueeze(0)@(K[1:])).squeeze()*h
    k_prods = b.unsqueeze(-1) * (K[1:])
    # print(f'k prods: {k_prods}')
    # print(f'ynew alt: {ynew.data}')

    # Calculate derivative of K's
    Kp = torch.zeros((order, yi.shape[-1], yi.shape[-1]))
    I = torch.eye(yi.shape[-1])
    for i in range(order):
        ks = torch.sum(A[i].unsqueeze(-1).unsqueeze(-1) * Kp * h, dim=0)
        kp = net_der(model, xi + c[i] * h, yi + dy[i])
        Kp[i] = kp @ (I + ks)
    # Combine K derivatives to get full derivative
    dyn = I + (b.unsqueeze(-1).unsqueeze(-1) * (Kp)).sum(dim=0)
    # print(f'dy:\n {dy}')
    return dyn, ynew.detach()


def ode_lyapunov(model, x_list, y_list, h=0.01, eps= 0.1, integrator='rk4'):
    cuda = next(model.parameters()).is_cuda
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seq_length, state_dim = y_list.shape
    rvals = torch.eye(state_dim).unsqueeze(0).repeat(seq_length, 1, 1).to(device)  # storage
    Q = torch.eye(state_dim)*eps
    ynew_list = []
    for i in range(seq_length):
        J, ynew = ode_jacobian(model, x_list[i], y_list[i], h=h, integrator=integrator)
        ynew_list.append(ynew)
        Q = torch.matmul(J, Q)
        qr_dict = oneStepVarQR(J, Q)
        Q, r = qr_dict['Q']*eps, qr_dict['R']/eps
        rvals[i] = r
    LEs = torch.sum(torch.log2(torch.diagonal(rvals.detach(), dim1=-2, dim2=-1)), dim=-2)/seq_length
    # print(f'rval sum: {torch.log2(rvals.diagonal(dim1=-2, dim2=-1)).sum(dim=-2)}')
    return LEs, rvals, torch.stack(ynew_list)


def oneStepVarQR(J, Q):
    Z = torch.matmul(torch.transpose(J, -2, -1), Q)  # Linear extrapolation of the network in many directions
    q, r = torch.linalg.qr(Z, mode='reduced')  # QR decomposition of new directions
    s = torch.diag_embed(torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)))  # extract sign of each leading r value
    return {'Q': torch.matmul(q, s), 'R': torch.matmul(s, r),
            'S': s}  # return positive r values and corresponding vectors
