import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

import ot

def sinkhorn(a, b, M, reg, numItermax = 1000, stopThr = 1e-9, verbose = False):
    """
    Solve the entropic regularization balanced optimal transport problem 

    Parameters:
    param: a(tensor (I, )) sample weights for source measure
    param: b(tensor (J, )) sample weights for target measure
    param: M(tensor (I, J)) distance matrix between source and target measure
    param: reg(float64) regularization factor > 0
    param: numItermax(int) max number of iterations
    param: stopThr(float64) stop threshol
    param: verbose(bool) print information along iterations

    Return:
    P(tensor (I, J)) the final transport plan
    loss(float) the wasserstein distance between source and target measure
    """
    import time
    assert a.device == b.device and b.device == M.device, "a, b, M must be on the same device"

    device = a.device
    a, b, M = a.type(torch.DoubleTensor).to(device), b.type(torch.DoubleTensor).to(device), M.type(torch.DoubleTensor).to(device)

    if len(a) == 0:
        a = torch.ones(M.shape[0], dtype=torch.DoubleTensor) / M.shape[0]
    if len(b) == 0:
        b = torch.ones(M.shape[1], dtype=torch.DoubleTensor) / M.shape[1]
    
    I, J = len(a), len(b)
    assert I == M.shape[0] and J == M.shape[1], "the dimension of weights and distance matrix don't match"

    # init 
    u = torch.ones((I, 1), device = device, dtype=a.dtype) / I
    v = torch.ones((J, 1), device = device, dtype=b.dtype) / J
    # K = torch.exp(-M / reg).to(device)
    K = torch.empty(M.size(), dtype=M.dtype, device=device)
    torch.div(M, -reg, out=K)
    torch.exp(K, out=K)

    tmp2 = torch.empty(b.shape, dtype=b.dtype, device=device)

    Kp = (1 / a).reshape(-1, 1) * K
    cpt, err = 0, 1 
    # pos = time.time()
    while (err > stopThr and cpt < numItermax):
        uprev, vprev = u, v

        KtranposeU = torch.mm(K.t(), u)
        v = b.reshape(-1, 1) / KtranposeU
        u = 1. / Kp.mm(v)

        if (torch.any(KtranposeU == 0)
                or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
            print("Warning: numerical errors at iteration ", cpt)
            u, v = uprev, vprev
            break
        
        if cpt % 10 == 0:
            tmp2 = torch.einsum('ia,ij,jb->j', u, K, v)
            err = torch.norm(tmp2 - b)
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:5s}'.format('It.','Err') + '\n' + '-' * 19)
                print("{:5s}|{:5s}".format(cpt, err))
        
        cpt += 1
    # print("ours cpt: {}, err: {}".format(cpt, err))
    # print("ours time: {}".format(time.time() - pos))
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    return P, torch.sum(P * M)


def Prior_sinkhorn(a, b, M, T, reg1, reg2, numItermax = 1000, stopThr = 1e-9, verbose = False):
    """
    Solve the entropic regularization balanced optimal transport problem 

    Parameters:
    param: a(tensor (I, )) sample weights for source measure
    param: b(tensor (J, )) sample weights for target measure
    param: M(tensor (I, J)) distance matrix between source and target measure
    param: T(tensor (I, J)) the prior transport plan of the problem
    param: reg1(float64) regularization factor > 0 for the enrtropic term
    param: reg2(float64) regularization factor > 0 for the KL divergence term between P and T
    param: numItermax(int) max number of iterations
    param: stopThr(float64) stop threshol
    param: verbose(bool) print information along iterations

    Return:
    P(tensor (I, J)) the final transport plan
    loss(float) the wasserstein distance between source and target measure
    """
    import time
    assert a.device == b.device and b.device == M.device, "a, b, M must be on the same device"

    device = a.device
    a, b, M = a.type(torch.DoubleTensor).to(device), b.type(torch.DoubleTensor).to(device), M.type(torch.DoubleTensor).to(device)

    if len(a) == 0:
        a = torch.ones(M.shape[0], dtype=torch.DoubleTensor) / M.shape[0]
    if len(b) == 0:
        b = torch.ones(M.shape[1], dtype=torch.DoubleTensor) / M.shape[1]
    
    I, J = len(a), len(b)
    assert I == M.shape[0] and J == M.shape[1], "the dimension of weights and distance matrix don't match"

    # init 
    u = torch.ones((I, 1), device = device, dtype=a.dtype) / I
    v = torch.ones((J, 1), device = device, dtype=b.dtype) / J

    # compute K 
    K1 = torch.empty(M.size(), dtype=M.dtype, device=device)
    torch.div(M, -(reg1 + reg2), out=K1)
    K2 = torch.empty(M.size(), dtype=M.dtype, device=device)
    torch.log(T, out= K2)
    K2 = K2 * reg2
    torch.div(K2, (reg1 + reg2), out=K2)
    K = torch.empty(M.size(), dtype=M.dtype, device=device)
    torch.exp(K1 + K2, out=K)

    tmp2 = torch.empty(b.shape, dtype=b.dtype, device=device)

    Kp = (1 / a).reshape(-1, 1) * K
    cpt, err = 0, 1 
    # pos = time.time()
    while (err > stopThr and cpt < numItermax):
        uprev, vprev = u, v

        KtranposeU = torch.mm(K.t(), u)
        v = b.reshape(-1, 1) / KtranposeU
        u = 1. / Kp.mm(v)

        if (torch.any(KtranposeU == 0)
                or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
            print("Warning: numerical errors at iteration ", cpt)
            u, v = uprev, vprev
            break
        
        if cpt % 10 == 0:
            tmp2 = torch.einsum('ia,ij,jb->j', u, K, v)
            err = torch.norm(tmp2 - b)
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:5s}'.format('It.','Err') + '\n' + '-' * 19)
                print("{:5s}|{:5s}".format(cpt, err))
        
        cpt += 1
    # print("ours cpt: {}, err: {}".format(cpt, err))
    # print("ours time: {}".format(time.time() - pos))
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    return P, torch.sum(P * M)

def VGW(mu, nu, X, Y, reg, numItermax = 1000, stopThr = 1e-9, verbose = False):
    assert X.shape[0] == Y.shape[0]
    C1_square = (X * X).sum(axis=1, keepdim=True) - 2 * X.mm(X.t()) + (X * X).sum(axis=1, keepdim=True).reshape(1, -1)
    C2_square = (Y * Y).sum(axis=1, keepdim=True) - 2 * Y.mm(Y.t()) + (Y * Y).sum(axis=1, keepdim=True).reshape(1, -1)
    Xmean = (mu.reshape(1, -1).mm(X)).reshape(1, -1)
    Ymean = (nu.reshape(1, -1).mm(Y)).reshape(-1, 1)
    E = (C1_square * mu.reshape(-1, 1)).sum(axis = 0).reshape(-1, 1) + (C2_square * nu.reshape(-1, 1)).sum(axis = 0).reshape(1, -1) + 2 * X.mm(Ymean) + 2 * Xmean.mm(Y.t())
    Mt = E - 4 * torch.eye(E.shape[0], device=X.device)
    P, _ = sinkhorn(mu, nu, Mt, reg, numItermax, stopThr, verbose)
    return P, _ 

if __name__ == "__main__":
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    k = 500
    X = torch.rand(k, 10)     
    Y = torch.rand(k, 10)
    # l2 ^ 2 distance matrix
    C = (X * X).sum(axis = 1, keepdim=True) + (Y * Y).sum(axis = 1) - 2 * (X.mm(Y.t())) 
    mu, nu = torch.ones(k), torch.ones(k)
    # mu, nu = torch.rand(k), torch.rand(k)
    # mu, nu = mu / torch.norm(mu), nu / torch.norm(nu)

    import ot.gpu
    import time
    mu_, nu_, C_ = mu.squeeze().numpy(), nu.squeeze().numpy(), C.numpy()
    pos = time.time()
    P_numpy = ot.gpu.sinkhorn(mu_, nu_, C_, 0.05, stopThr=1e-3)
    print("ot.gpu consume: {}".format(time.time() - pos))
    pos = time.time()
    P_torch, _ = sinkhorn(mu, nu, C, 0.05, stopThr=1e-3) 
    print("ours consume: {}".format(time.time() - pos))
    T = torch.rand_like(C, dtype=torch.float64)
    P_prior, _ = Prior_sinkhorn(mu, nu, C, T, 0.05, 0.05, stopThr=1e-3)
    print("error between P_numpy and P_torch: {}".format(np.linalg.norm(P_numpy - P_torch.numpy())))
    print("error between P_prior and P_torch: {}".format(np.linalg.norm(P_prior.numpy() - P_torch.numpy())))