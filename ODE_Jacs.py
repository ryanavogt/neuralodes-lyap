import numpy as np
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def net_der(model, xi, yi):
    W = model[0].weight
    b = model[0].bias
    Wout = model[-1].weight
    bout = model[-1].bias
    arg = yi @ W.T + b
    der = (1 - torch.tanh(arg) ** 2).unsqueeze(1) * W.T.unsqueeze(0)
    # print(der.shape)
    der_out = der @ Wout.T
    # print(f'der out shape: {der_out.shape}')
    return der_out


def numerical_der(model, xi, yi, eps= 0.01):
    n_dims = yi.shape[-1]
    delta= eps*torch.eye(n_dims)
    with torch.no_grad():
        y_init= yi.unsqueeze(0).repeat(n_dims,1)
        yi_fwd = y_init + delta
        yi_bwd = y_init - delta
        print(f'yi {yi}')
        print(f'yi_fwd {yi_fwd}')
        fwd = model(xi, yi_fwd)
        bwd = model(xi, yi_bwd)
        jac = (fwd-bwd)/(2*eps)
    return jac


def nonlinear_pendulum(y, omega_0=torch.sqrt(torch.Tensor([9.81])).double().to(device)):
    dtheta = y[:, -1].unsqueeze(-1)
    # dtheta = torch.hstack([torch.zeros_like(dtheta), dtheta])
    dtheta = torch.hstack([torch.zeros_like(dtheta), torch.ones_like(dtheta)])
    dv = -omega_0 ** 2 * torch.cos(y[:, 0]).double().unsqueeze(-1)
    # dv = -omega_0 ** 2 * torch.sin(y[:, 0]).double().unsqueeze(-1)
    dv = torch.hstack([dv, torch.zeros_like(dv)])
    J = torch.dstack([dtheta, dv])
    # print(f'dy shape: {dy.shape}')
    return J


def simple_pendulum(y, omega_0 = torch.sqrt(torch.Tensor([9.81])).double().to(device)):
    dtheta = y[:, -1].unsqueeze(-1)
    # dtheta = torch.hstack([torch.zeros_like(dtheta), dtheta])
    dtheta = torch.hstack([torch.zeros_like(dtheta), torch.ones_like(dtheta)])
    # dv = -2* omega_0 ** 2 * y[:, 0].unsqueeze(-1)
    dv = -(omega_0 ** 2) * torch.ones_like(y[:, 0]).unsqueeze(-1)
    dv = torch.hstack([dv, torch.zeros_like(dv)])
    J = torch.dstack([dtheta, dv])
    return J


def rk_alt(model, xi, yi, h=0.01):
    k1 = model(0, yi)
    k2 = model(0, yi + h * k1 * 1./3)
    k3 = model(0, yi + h * (k2 - k1 * 1./3))
    k4 = model(0, yi + h * (k1 - k2 + k3))
    ynew = (k1 + 3 * (k2 + k3) + k4) * h * 0.125
    return ynew


def integrate_step(model, yi, xi=0, h=0.01, integrator='rk4'):
    if integrator == 'rk4_38':
        order = 4
        A = torch.Tensor([[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]])
        b = torch.Tensor([1 / 8, 3 / 8, 3 / 8, 1 / 8])
        c = torch.Tensor([0, 1 / 3, 2 / 3, 1])

    if integrator == 'rk4':
        order = 4
        A = torch.Tensor([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]])
        b = torch.Tensor([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        c = torch.Tensor([0, 1 / 2, 1 / 2, 1])

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

    batch_size = yi.shape[0]
    dims = yi.shape[-1]

    # Calculate quantities for integrator
    # print(f'yi shape: {yi.shape}')
    K = torch.zeros((order, batch_size, dims))
    dy = torch.zeros((order, batch_size, dims))

    # Calculate K's of integration method
    for i in range(order):
        # print(f'A Shape: {A[i].unsqueeze(0).shape}')
        # print(f'K Shape: {K[1:].shape}')
        dy[i] = (A[i].unsqueeze(0) @ K.transpose(0, 1)).sum(dim=1)*h
        k = model(yi + dy[i])
        K[i] = k
    # print(f'bprod shape: {(b.unsqueeze(0)@(K[1:].transpose(0,1))).shape}')
    ynew = yi + (b.unsqueeze(0)@(K.transpose(0,1))).squeeze()*h

    return ynew


def ode_jacobian(model, yi, xi=0, h=0.01, integrator='rk4'):
    if integrator == 'rk4_38':
        order = 4
        A = torch.Tensor([[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]])
        b = torch.Tensor([1 / 8, 3 / 8, 3 / 8, 1 / 8])
        c = torch.Tensor([0, 1 / 3, 2 / 3, 1])

    elif integrator == 'rk4':
        order = 4
        A = torch.Tensor([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]])
        b = torch.Tensor([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        c = torch.Tensor([0, 1 / 2, 1 / 2, 1])

    elif integrator == 'euler':
        order = 1
        A=torch.Tensor([[0]])
        b=torch.Tensor([1])
        c=torch.Tensor([0])

    elif integrator == 'midpoint':
        order = 2
        A = torch.Tensor([[0, 0], [1/2, 0]])
        b = torch.Tensor([0, 1])
        c = torch.Tensor([0, 1/2])

    # Calculate quantities for integrator
    batch_size = yi.shape[0]
    dims = yi.shape[-1]

    # Calculate quantities for integrator
    # print(f'yi shape: {yi.shape}')
    K = torch.zeros((order, batch_size, dims)).to(device)
    dy = torch.zeros((order, batch_size, dims)).to(device)

    # Calculate K's of integration method
    for i in range(order):
        dy[i] = (A[i].unsqueeze(0) @ K.transpose(0, 1)).sum(dim=1)*h
        k = model(yi + dy[i])
        K[i] = k
    ynew = yi + (b.unsqueeze(0)@(K.transpose(0,1))).squeeze()*h

    # Calculate derivative of K's
    Kp = torch.zeros((order, batch_size, dims, dims))
    I = torch.eye(dims).unsqueeze(0).repeat(batch_size, 1, 1).float()
    for i in range(order):
        ks = torch.sum(A[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * Kp.transpose(0,1) * h, dim=1).float()
        kp = net_der(model, xi + c[i] * h, yi + dy[i]).float()
        Kp[i] = kp @ (I + ks)
    # Combine K derivatives to get full derivative
    dyn = I + (b.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * (Kp.transpose(0,1))).sum(dim=1)
    # print(f'dy:\n {dy}')
    return dyn


def ode_lyapunov(model, y_list, h=0.01, eps= 0.1, integrator='rk4'):
    cuda = next(model.parameters()).is_cuda
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seq_length, state_dim = y_list.shape
    rvals = torch.eye(state_dim).unsqueeze(0).repeat(seq_length, 1, 1).to(device)  # storage
    Q = torch.eye(state_dim).to(device)*eps

    with torch.no_grad():
        # First, warmup to get eigenvectors to converge
        J= ode_jacobian(model, y_list.to(device), h=h, integrator=integrator)
        # ynew_list.append(ynew)
        for i in range(seq_length):
            qr_dict = oneStepVarQR(J[i], Q)
            Q, r = qr_dict['Q']*eps, qr_dict['R']/eps

        # Then, calculate evolution and track R values simultaneously
        J = ode_jacobian(model, y_list, h=h, integrator=integrator)
        # ynew_list.append(ynew)
        for i in range(seq_length):
            qr_dict = oneStepVarQR(J[i], Q)
            Q, r = qr_dict['Q']*eps, qr_dict['R']/eps
            rvals[i] = r
    LEs = torch.sum(torch.log2(torch.diagonal(rvals.detach(), dim1=-2, dim2=-1)), dim=-2)/seq_length
    # print(f'rval sum: {torch.log2(rvals.diagonal(dim1=-2, dim2=-1)).sum(dim=-2)}')
    return LEs, rvals, J


def dysys_lyapunov(y_list, h=0.01, eps=1, system='dy_sys', alt = False):
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if system == 'dy_sys':
        f = nonlinear_pendulum
    if system == 'simple':
        f = simple_pendulum
    seq_length, state_dim = y_list.shape
    qvects = []
    rvals = torch.eye(state_dim).unsqueeze(0).repeat(seq_length, 1, 1).to(device)  # storage
    Q = torch.eye(state_dim).double().to(device) * eps

    if alt:
        J = f(y_list)
        X = torch.exp(J)
        U, S, V = torch.linalg.svd(X.transpose(-2, -1)@X)
        LEs = 1/torch.arange(X.shape[0])*torch.log2(torch.sqrt(S))

    else:
        with torch.no_grad():
            # First, warmup to get eigenvectors to converge
            J = f(y_list)
            # ynew_list.append(ynew)
            for i in range(seq_length):
                # Q = torch.matmul(J[i], Q)
                qr_dict = oneStepVarQR(J[i], Q)
                Q, r = qr_dict['Q'] * eps, qr_dict['R']

            # Then, calculate evolution and track R values simultaneously
            J = f(y_list)
            # ynew_list.append(ynew)
            for i in range(seq_length):
                # Q = torch.matmul(J[i], Q)
                qr_dict = oneStepVarQR(J[i], Q)
                Q, r = qr_dict['Q'] * eps, qr_dict['R']/eps
                qvects.append(Q)
                rvals[i] = r
        qvects = torch.vstack(qvects)
        LEs = torch.sum(torch.log2(torch.diagonal(rvals.detach(), dim1=-2, dim2=-1)), dim=-2) / seq_length
    # print(f'rval sum: {torch.log2(rvals.diagonal(dim1=-2, dim2=-1)).sum(dim=-2)}')
    return LEs, rvals, J


def oneStepVarQR(J, Q):
    Z = torch.matmul(torch.transpose(J, -2, -1), Q)  # Linear extrapolation of the network in many directions
    q, r = torch.linalg.qr(Z, mode='reduced')  # QR decomposition of new directions
    s = torch.diag_embed(torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)))  # extract sign of each leading r value
    return {'Q': torch.matmul(q, s), 'R': torch.matmul(s, r),
            'S': s}  # return positive r values and corresponding vectors
