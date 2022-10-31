import numpy as np
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def net_der(model, xi, yi):
    W = model[0].weight
    b = model[0].bias
    Wout = model[-1].weight
    bout = model[-1].bias
    arg = (yi @ W.T + b).numpy()
    der = np.expand_dims(1 - np.tanh(arg) ** 2, axis=1) * (W.T.unsqueeze(0)).numpy()
    # print(der.shape)
    der_out = der @ (Wout.T).numpy()
    # print(f'der out shape: {der_out.shape}')
    return der_out


def numerical_der(model, xi, yi, eps=0.01):
    n_dims = yi.shape[-1]
    delta = eps * np.eye(n_dims)
    with torch.no_grad():
        y_init = yi.unsqueeze(0).repeat(n_dims, 1)
        yi_fwd = y_init + delta
        yi_bwd = y_init - delta
        print(f'yi {yi}')
        print(f'yi_fwd {yi_fwd}')
        fwd = model(xi, yi_fwd)
        bwd = model(xi, yi_bwd)
        jac = (fwd - bwd) / (2 * eps)
    return jac


def nonlinear_pendulum(y, h=0.1, omega_0=np.sqrt(np.array([9.81]))):
    dtheta = np.expand_dims(y[:, -1].cpu().numpy(), axis=-1)
    # dtheta = torch.hstack([torch.zeros_like(dtheta), dtheta])
    dtheta = np.hstack([np.zeros_like(dtheta), np.ones_like(dtheta)])
    dv = -omega_0 ** 2 * np.expand_dims(np.cos(y[:, 0].cpu().numpy()), axis=1)
    # dv = -omega_0 ** 2 * torch.sin(y[:, 0]).double().unsqueeze(-1)
    dv = np.hstack([dv, np.zeros_like(dv)])
    J = np.dstack([dtheta, dv])
    # return J
    return np.eye(J.shape[-1]) + h * J


def simple_pendulum(y, h=0.1, omega_0=np.sqrt(np.array([9.81]))):
    dtheta = np.expand_dims(y[:, -1].cpu().numpy(), axis=-1)
    # dtheta = torch.hstack([torch.zeros_like(dtheta), dtheta])
    dtheta = np.hstack([np.zeros_like(dtheta), np.ones_like(dtheta)])
    dv = -(omega_0 ** 2) * np.expand_dims(np.ones_like(y[:, 0].cpu().numpy()), axis=-1)
    dv = np.hstack([dv, np.zeros_like(dv)])
    J = np.dstack([dtheta, dv])
    # return np.eye(J.shape[-1]) + h*J
    return J


def rk_alt(model, xi, yi, h=0.01):
    k1 = model(0, yi)
    k2 = model(0, yi + h * k1 * 1. / 3)
    k3 = model(0, yi + h * (k2 - k1 * 1. / 3))
    k4 = model(0, yi + h * (k1 - k2 + k3))
    ynew = (k1 + 3 * (k2 + k3) + k4) * h * 0.125
    return ynew


def integrate_step(model, yi, xi=0, h=0.01, integrator='rk4'):
    if integrator == 'rk4_38':
        order = 4
        A = np.array([[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]])
        b = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8])
        c = np.array([0, 1 / 3, 2 / 3, 1])

    if integrator == 'rk4':
        order = 4
        A = np.array([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]])
        b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        c = np.array([0, 1 / 2, 1 / 2, 1])

    if integrator == 'euler':
        order = 1
        A = np.array([[0]])
        b = np.array([1])
        c = np.array([0])

    if integrator == 'midpoint':
        order = 2
        A = np.array([[0, 0], [1 / 2, 0]])
        b = np.array([0, 1])
        c = np.array([0, 1 / 2])

    batch_size = yi.shape[0]
    dims = yi.shape[-1]

    # Calculate quantities for integrator
    # print(f'yi shape: {yi.shape}')
    K = np.zeros((order, batch_size, dims))
    dy = np.zeros((order, batch_size, dims))

    # Calculate K's of integration method
    for i in range(order):
        # print(f'A Shape: {A[i].unsqueeze(0).shape}')
        # print(f'K Shape: {K[1:].shape}')
        dy[i] = (A[i].unsqueeze(0) @ K.transpose(0, 1)).sum(dim=1) * h
        k = model(yi + dy[i])
        K[i] = k
    # print(f'bprod shape: {(b.unsqueeze(0)@(K[1:].transpose(0,1))).shape}')
    ynew = yi + (b.unsqueeze(0) @ (K.transpose(0, 1))).squeeze() * h

    return ynew


def ode_jacobian(model, yi, xi=0, h=0.01, integrator='rk4'):
    if integrator == 'rk4_38':
        order = 4
        A = np.array([[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]])
        b = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8])
        c = np.array([0, 1 / 3, 2 / 3, 1])

    elif integrator == 'rk4':
        order = 4
        A = np.array([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]])
        b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        c = np.array([0, 1 / 2, 1 / 2, 1])

    elif integrator == 'euler':
        order = 1
        A = np.array([[0]])
        b = np.array([1])
        c = np.array([0])

    elif integrator == 'midpoint':
        order = 2
        A = np.array([[0, 0], [1 / 2, 0]])
        b = np.array([0, 1])
        c = np.array([0, 1 / 2])

    # Calculate quantities for integrator
    batch_size = yi.shape[0]
    dims = yi.shape[-1]

    # Calculate quantities for integrator
    # print(f'yi shape: {yi.shape}')
    K = np.zeros((order, batch_size, dims))
    dy = np.zeros((order, batch_size, dims))

    # Calculate K's of integration method
    for i in range(order):
        dy[i] = (A[i].unsqueeze(0) @ K.transpose(0, 1)).sum(dim=1) * h
        k = model(yi + dy[i])
        K[i] = k
    ynew = yi + (b.unsqueeze(0) @ (K.transpose(0, 1))).squeeze() * h

    # Calculate derivative of K's
    Kp = torch.zeros((order, batch_size, dims, dims))
    I = torch.eye(dims).unsqueeze(0).repeat(batch_size, 1, 1).float()
    for i in range(order):
        ks = np.sum(A[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * Kp.transpose(0, 1) * h, axis=1)
        kp = net_der(model, xi + c[i] * h, yi + dy[i])
        Kp[i] = kp @ (I + ks)
    # Combine K derivatives to get full derivative
    dyn = I + h * (b.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * (Kp.transpose(0, 1))).sum(axis=1)
    # Should an additional h be added to this expression?
    # print(f'dy:\n {dy}')
    return dyn


def ode_lyapunov(model, y_list, h=0.01, eps=0.1, integrator='rk4'):
    cuda = next(model.parameters()).is_cuda
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seq_length, state_dim = y_list.shape
    rvals = torch.eye(state_dim).unsqueeze(0).repeat(seq_length, 1, 1).to(device)  # storage
    Q = torch.eye(state_dim).to(device) * eps

    with torch.no_grad():
        # First, warmup to get eigenvectors to converge
        J = ode_jacobian(model, y_list.to(device), h=h, integrator=integrator)
        # ynew_list.append(ynew)
        for i in range(seq_length):
            qr_dict = oneStepVarQR(J[i], Q)
            Q, r = qr_dict['Q'] * eps, qr_dict['R'] / eps

        # Then, calculate evolution and track R values simultaneously
        J = ode_jacobian(model, y_list, h=h, integrator=integrator)
        # ynew_list.append(ynew)
        for i in range(seq_length):
            qr_dict = oneStepVarQR(J[i], Q)
            Q, r = qr_dict['Q'] * eps, qr_dict['R'] / eps
            rvals[i] = r
    LEs = torch.sum(torch.log2(torch.diagonal(rvals.detach(), dim1=-2, dim2=-1)), dim=-2) / seq_length
    # print(f'rval sum: {torch.log2(rvals.diagonal(dim1=-2, dim2=-1)).sum(dim=-2)}')
    return LEs, rvals, J


def dysys_lyapunov(y_list, h=0.01, eps=1, system='dy_sys', alt=False):
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if system == 'dy_sys':
        f = nonlinear_pendulum
        eq = pendulum
    if system == 'simple':
        f = simple_pendulum
        eq = linear
    seq_length, state_dim = y_list.shape
    qvects = []
    rvals = np.expand_dims(np.eye(state_dim), axis=0).repeat(seq_length, axis=0)  # storage
    Q = np.eye(state_dim)

    if alt:
        le_dict = lyapunovLeapfrog(eq, y_list, h)
        LEs = le_dict['LEs']
        J = np.zeros((seq_length, state_dim, state_dim))

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
            J = f(y_list, h)
            # ynew_list.append(ynew)
            for i in range(seq_length):
                # Q = torch.matmul(J[i], Q)
                qr_dict = oneStepVarQR(J[i], Q)
                Q, r = qr_dict['Q'] * eps, qr_dict['R'] / eps
                qvects.append(Q)
                rvals[i] = r
        qvects = np.vstack(qvects)
        LEs = np.sum(np.log2(np.diagonal(rvals, axis1=-2, axis2=-1)), axis=-2) / (seq_length)
    # print(f'rval sum: {torch.log2(rvals.diagonal(dim1=-2, dim2=-1)).sum(dim=-2)}')
    return LEs, rvals, J


def oneStepVarQR(J, Q):
    Z = np.matmul(J, Q)  # Linear extrapolation of the network in many directions
    q, r = np.linalg.qr(Z, mode='reduced')  # QR decomposition of new directions
    s = torch.diag_embed(
        torch.DoubleTensor(np.sign(np.diagonal(r, axis1=-2, axis2=-1)))).numpy()  # extract sign of each leading r value
    return {'Q': np.matmul(q, s), 'R': np.matmul(s, r),
            'S': s}  # return positive r values and corresponding vectors


def pendulum(x, omega_0=np.sqrt(9.81)):
    return -omega_0 ** 2 * np.sin(x)


def linear(x, omega_0=np.sqrt(9.81)):
    return -omega_0 ** 2 * x


def lyapunovLeapfrog(f, y_list, dt):
    print("Using Leapfrog")
    seq_length, state_dim = y_list.shape
    Q = np.eye(state_dim)
    Xi_list = []
    rvals = []
    for y in y_list:
        Xp = (np.expand_dims(y, axis=-1) + Q)
        x0 = Xp[:, 0]
        v0 = Xp[:, 1]
        a0 = f(x0)
        v1 = v0 + a0 * dt / 2
        xf = x0 + v1 * dt
        af = f(xf)
        vf = v1 + af * dt / 2
        Xi = np.vstack([xf, vf])
        Xi_list.append(Xi)
        Z = Xi-np.expand_dims(y, axis=-1)
        q, r = np.linalg.qr(Z, mode='reduced')  # QR decomposition of new directions
        s = torch.diag_embed(
            torch.DoubleTensor(np.sign(np.diagonal(r, axis1=-2, axis2=-1)))).numpy()
        Q = np.matmul(q , s)
        rvals.append(np.matmul(s , r))
    # Z = np.vstack([xf, vf])
    rvals = np.stack(rvals)
    LEs = np.sum(np.log2(np.diagonal(rvals, axis1=-2, axis2=-1)), axis=-2) / seq_length
    return {'LEs': LEs, 'R': rvals}  # return positive r values and corresponding vectors
