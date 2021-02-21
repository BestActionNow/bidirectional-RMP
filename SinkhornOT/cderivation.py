import math

import torch
from torch.nn import functional as F

big = 1e20
huge = 1e30
small = 1e-7


######################
# Euclidean distance #
######################
def p_norm_dist_mat(x, y, p=2):
    """
    Euclidean norm matrix
    :param x: source vectors, [n1, d]
    :param y: target vectors, [n2, d]
    :return: a distance metric, [n1, n2]
    """
    n1, d1 = x.shape
    n2, d2 = y.shape
    assert d1 == d2
    D = torch.sum(torch.pow(x.reshape(n1, 1, d1) -
                            y.reshape(1, n2, d2), p), -1)
    return D


def norm_dist_mat(x, y, p=2):
    """
    Euclidean norm matrix
    :param x: source vectors, [n1, d]
    :param y: target vectors, [n2, d]
    :return: a distance metric, [n1, n2]
    """
    D = torch.pow(p_norm_dist_mat(x, y, p), 1 / p)
    return D


##################
# Cos similarity #
##################

def cos_dist_mat(x, y):
    """
    Cos similarity
    :param x: [n1, d]
    :param y: [n2, d]
    :return: [n1, n2]
    """
    n1, d1 = x.shape
    n2, d2 = y.shape
    assert d1 == d2

    if n1 * n2 > 10000:
        D = torch.cat(
            [torch.cosine_similarity(x[i].view(1, 1, d1), y.view(1, n2, d2), -1) for i in range(n1)], 0
        )
    else:
        D = torch.cosine_similarity(x.reshape(n1, 1, d1), y.reshape(1, n2, d2), -1)
    return 1 - D


###################
# Energy Distance #
###################
def energy_padding(x, y):
    """
    padding two input vector list (matrix) as two sample and their arrangement
    :param x: input matrix 1 [2 * n1, d1]
    :param y: input matrix 2 [2 * n2, d2]
    :return: z1, z2
    """
    n1, d1 = x.shape
    n2, d2 = y.shape
    assert d1 == d2
    assert n1 == n2
    assert n1 % 2 == 0
    assert n2 % 2 == 0

    x_1 = x[0: n1 // 2]
    x_1 = x_1.reshape(1, *x_1.shape)
    x_2 = x[n1 // 2: n1]
    x_2 = x_2.reshape(1, *x_2.shape)

    y_1 = y[0: n2 // 2]
    y_1 = y_1.reshape(1, *y_1.shape)
    y_2 = y[n2 // 2: n2]
    y_2 = y_2.reshape(1, *y_2.shape)

    z_1 = torch.cat((x_1, x_1, x_2, x_2, x_1, y_1),
                    0).reshape(6, n1 // 2, 1, d1)
    z_2 = torch.cat((y_1, y_2, y_1, y_2, x_2, y_2),
                    0).reshape(6, 1, n2 // 2, d2)
    return z_1, z_2


def square_energy_l2_dist_mat(x, y):
    """
    Matrix for energy distance tensor for optimal transport
    split the x and y and then rearrange them
    calculate sum_i (x_i - y_i)^2
    :param x: source vectors [2 * n1, d]
    :param y: target vectors [2 * n1, d]
    :return: a
    """
    z_1, z_2 = energy_padding(x, y)
    # TODO: Reduce the memory if necessary.
    #  see this link
    #  https://discuss.pytorch.org/t/all-to-all-element-wise-l2-distance-of-vector/4589
    D = torch.sum(torch.pow(z_1 - z_2, 2), -1)
    return D


def energy_l2_dist_mat(x, y):
    D = torch.sqrt(square_energy_l2_dist_mat(x, y))
    return D


def square_energy_cos_dist_mat(x, y):
    z_1, z_2 = energy_padding(x, y)
    z_1 = z_1.repeat(1, 1, z_2.shape[1], 1)
    z_2 = z_2.repeat(1, z_1.shape[1], 1, 1)
    cos_sim = F.cosine_similarity(z_1, z_2, dim=-1)
    D = F.relu(1 - cos_sim)
    return D


def energy_cos_dist_mat(x, y):
    D = torch.sqrt(square_energy_cos_dist_mat(x, y))
    return D


#####################################
# Common used in Gromov-Wasserstein #
#####################################
def get_intra_sim(x, sim_func):
    x = x.detach()
    return sim_func(x, x)


def get_inter_sim(x, y, sim_func):
    x, y = x.detach(), y.detach()
    return sim_func(x, y)


def get_init_matrices(C1, C2, mu, nu, div_type='l2'):
    I, *_ = C1.shape
    J, *_ = C2.shape
    _mu = mu.view(I, 1)
    _nu = nu.view(1, J)

    A = .5 * torch.matmul(C1 ** 2, _mu.repeat(1, J))
    B = .5 * torch.matmul(_nu.repeat(I, 1), C2.t() ** 2)
    constC = A + B
    hC1 = C1
    hC2 = C2
    return constC, hC1, hC2


def get_LT(constC, hC1, hC2, T):
    ten = torch.matmul(hC1, torch.matmul(T, hC2.t()))
    return constC - ten


#############################
# Construct the cost matrix #
#############################
def w2_cost_matrix(D, device):  # the cost function C of W2 metric
    return torch.pow(D, 2).to(device)


def wfr_cost_matrix(D, diameter, device):  # the cost function C of WFR metric
    pi2 = torch.tensor(math.pi / 2).to(device)
    # 1e-5 -> small = 1e-10 ?
    return -2 * torch.log(
        torch.cos(torch.min(torch.div(D, diameter) * pi2, pi2)) + small)


def GW_cost_matrix(constC, hC1, hC2, T_old, epsilon):
    lt = get_LT(constC, hC1, hC2, T_old)
    Dip = lt - epsilon * torch.log(T_old + small)
    return lt, Dip


def FGW_cost_matrix(D, constC, hC1, hC2, T, alpha, epsilon, p):
    A = (1 - alpha) * D ** p + alpha * get_LT(constC, hC1, hC2, T) ** p
    Dip = A - epsilon * torch.log(T)
    return A, Dip
