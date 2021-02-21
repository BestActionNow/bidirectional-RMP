import torch
from .cderivation import GW_cost_matrix, FGW_cost_matrix, get_init_matrices
from .sinkhorn_loss import sinkhorn_iteration, gsinkhorn_iteration, forward_relax_sinkhorn_iteration


def iterative_1(C1, C2, mu, nu, epsilon, max_iter, log, tol=1e-9, g=True, cost_mat_func=GW_cost_matrix, lambdda=0):
    I = C1.shape[0]
    J = C2.shape[0]
    dtype = C1.dtype
    device1 = C1.device
    device2 = C2.device
    assert device1 == device2

    mu = mu.view(1, I, 1)
    nu = nu.view(1, 1, J)

    T_old = torch.ones(I, J) / (I * J)
    T_old = T_old.to(device1).to(dtype)

    # note that for pc sinkhorn iteration
    constC, hC1, hC2 = get_init_matrices(C1, C2, mu, nu)

    lt, Dip = cost_mat_func(constC, hC1, hC2, T_old, epsilon)

    gw_dist = torch.sum(torch.mul(T_old, lt))

    if log:
        log = {'constC': constC.cpu().numpy(), 'hC1': hC1.cpu().numpy(), 'hC2': hC2.cpu().numpy(),
               'err': [], 'gwd': [], 'D': [], 'T': []}

    for i_proj in range(max_iter):
        if g:
            gw_dist, *_, T = forward_relax_sinkhorn_iteration(2 * lt.view(1, I, J), mu, nu, lambdda, epsilon)
        else:
            gw_dist, *_, T = sinkhorn_iteration(2 * lt.view(1, I, J), mu, nu, epsilon)
        err = torch.norm(T_old - T)

        if i_proj % 1 == 0:

            if log:
                log['err'].append(err.cpu().numpy())
                log['gwd'].append(gw_dist.cpu().numpy())
                log['T'].append(T.cpu().numpy())
                log['D'].append(2 * lt.cpu().numpy())
                print("Iteration:{:4d} err:{:.8f} gwd:{:.8f}".format(i_proj,err.item(),gw_dist.item()), end='\r')
            if err < tol:
                # print("meet tol jump out GW iteration")
                break

        T_old = T
        lt, Dip = GW_cost_matrix(constC, hC1, hC2, T_old, epsilon)
    print()
    if log:
        log['gw_dist'] = gw_dist.cpu().numpy() / 2
        return T, log
    else:
        return T, gw_dist

# TODO: fix the bugs of this iterative solver
def iterative_2(C1, C2, mu, nu, epsilon, max_iter, log, tol=1e-9, g=True, cost_mat_func=GW_cost_matrix, lambdda=0):
    I = C1.shape[0]
    J = C2.shape[0]
    dtype = C1.dtype
    device1 = C1.device
    device2 = C2.device
    assert device1 == device2

    mu = mu.view(1, I, 1)
    nu = nu.view(1, 1, J)

    T_old = torch.ones(I, J) / (I * J)
    T_old = T_old.to(device1).to(dtype)

    # note that for pc sinkhorn iteration
    constC, hC1, hC2 = get_init_matrices(C1, C2, mu, nu)

    lt, Dip = cost_mat_func(constC, hC1, hC2, T_old, epsilon)

    gw_dist = torch.sum(torch.mul(T_old, lt))

    if log:
        log = {'constC': constC.cpu().numpy(), 'hC1': hC1.cpu().numpy(), 'hC2': hC2.cpu().numpy(),
               'err': [], 'gwd': [], 'D': [], 'T': []}

    for i_proj in range(max_iter):

        if g:
            gw_dist, *_, T = gsinkhorn_iteration(Dip.view(1, I, J), mu, nu, lambdda, epsilon)
        else:
            gw_dist, *_, T = sinkhorn_iteration(Dip.view(1, I, J), mu, nu, epsilon)

        err = torch.norm(T_old - T)

        if i_proj % 1 == 0:

            if log:
                log['err'].append(err.cpu().numpy())
                log['gwd'].append(gw_dist.cpu().numpy())
                log['T'].append(T.cpu().numpy())
                log['D'].append(2 * lt.cpu().numpy())

            if err < tol:
                print("meet tol jump out GW iteration")
                break

        T_old = T
        lt, Dip = GW_cost_matrix(constC, hC1, hC2, T_old, epsilon)

    if log:
        log['gw_dist'] = gw_dist.cpu().numpy() / 2
        return T, log
    else:
        return T, gw_dist


def gw_iterative_1(C1, C2, mu, nu, epsilon, max_iter, log=False, tol=1e-9):
    return iterative_1(C1, C2, mu, nu, epsilon, max_iter, log, tol, False, GW_cost_matrix)


def rgw_iterative_1(C1, C2, mu, nu, max_iter, lambdda, epsilon, log=False, tol=1e-6):
    return iterative_1(C1, C2, mu, nu, epsilon, max_iter, log, tol, True, GW_cost_matrix, lambdda)


def fgw_iterative_1(D, C1, C2, mu, nu, alpha, p, max_iter, epsilon, log=False, tol=1e-6):
    def cost_matrix_func(constC, hC1, hC2, T, epsilon):
        return FGW_cost_matrix(D, constC, hC1, hC2, T, alpha, epsilon, p)
    return iterative_1(C1, C2, mu, nu, epsilon, max_iter, log, tol, False, cost_matrix_func)


def rfgw_iterative_1(D, C1, C2, mu, nu, alpha, p, max_iter, lambdda, epsilon, log=False, tol=1e-6):
    def cost_matrix_func(constC, hC1, hC2, T, epsilon):
        return FGW_cost_matrix(D, constC, hC1, hC2, T, alpha, epsilon, p)
    return iterative_1(C1, C2, mu, nu, epsilon, max_iter, log, tol, True, cost_matrix_func, lambdda)


# TODO: FIX the iterative projection
# def gw_iterative_2(C1, C2, mu, nu, epsilon, max_iter, log, tol=1e-9):
#     return iterative_2(C1, C2, mu, nu, epsilon, max_iter, log, tol, False, GW_cost_matrix)


# def rgw_iterative_2(C1, C2, mu, nu, max_iter, lambdda, epsilon, log, tol=1e-6):
#     return iterative_2(C1, C2, mu, nu, epsilon, max_iter, log, tol, True, GW_cost_matrix, lambdda)


# def fgw_iterative_2(D, C1, C2, mu, nu, alpha, p, max_iter, epsilon, log, tol=1e-6):
#     def cost_matrix_func(constC, hC1, hC2, T, epsilon):
#         return FGW_cost_matrix(D, constC, hC1, hC2, T, alpha, epsilon, p)
#     return iterative_2(C1, C2, mu, nu, epsilon, max_iter, log, tol, False, cost_matrix_func)


# def rfgw_iterative_2(D, C1, C2, mu, nu, alpha, p, max_iter, lambdda, epsilon, log, tol=1e-6):
#     def cost_matrix_func(constC, hC1, hC2, T, epsilon):
#         return FGW_cost_matrix(D, constC, hC1, hC2, T, alpha, epsilon, p)
#     return iterative_2(C1, C2, mu, nu, epsilon, max_iter, log, tol, True, cost_matrix_func, lambdda)

