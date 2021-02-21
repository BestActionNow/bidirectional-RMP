import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

import ot

big = 1e20
huge = 1e30
small = 1e-7
def myclamp(x):
    return torch.clamp(x, 0, huge)


default_device = "cuda:0" if torch.cuda.is_available() else "cpu"

def kl_div(x, y):
    """
    KL divergence of two tensor
    :param x:
    :param y:
    :return:
    """
    # y = y.reshape(x.shape)
    div = torch.div(x, y + small)  # avoid singularity
    kl = torch.mul(y, div * torch.log(div + small) - div + 1)
    return kl

# TODO: decouple the cost calculation and the sinkhorn
class WFRPointCloudCostFunc(Function):
    """
    WFR between two uniform point cloud with given cost matrix and delta
    """
    @staticmethod
    def forward(ctx, C, delta, epsilon=1e-2, niter=50, device=default_device):
        """
        Sinkhorn Iteration for WFR
        params:
            ctx: must have
            C: cost matrix [batch_size, I, J]
            delta: hyperparameter for WFR distance
            niter: number of iteration
            device: calculation device
        """
        batchSize, I, J = C.shape
        C = C.to(device)
        mu = torch.tensor([1.0] * I).to(device)
        nu = torch.tensor([1.0] * J).to(device)
        dx = torch.tensor([1 / I] * I).reshape([1, I, 1]).to(device)
        dy = torch.tensor([1 / J] * J).reshape([1, 1, J]).to(device)

        p_coef = 1 / (1 + epsilon)

        def K_calc(_u, _v):
            return torch.clamp(
                torch.exp((_u.view([-1, I, 1]) + _v.view([-1, 1, J] - C)) / epsilon), 0, huge)

        b = torch.ones_like(nu)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        K = K_calc(u, v)

        for ii in range(niter):
            b = b.reshape([-1, 1, J])
            bdy = torch.mul(b, dy)
            s = torch.sum(torch.mul(K, bdy), -1)
            a = torch.clamp(
                torch.pow(torch.div(mu, torch.exp(u) * s), p_coef), 0, huge)

            a = a.reshape([-1, I, 1])
            adx = torch.mul(a, dx)
            s = torch.sum(torch.mul(K, adx), -2)
            b = torch.clamp(
                torch.pow(torch.div(nu, torch.exp(v) * s), p_coef), 0, huge)

            if torch.max(a) > big or torch.max(b) > big or ii == niter - 1:
                u = epsilon * torch.log(a).squeeze() + u
                v = epsilon * torch.log(b).squeeze() + v
                K = K_calc(u, v)
                b = torch.ones_like(nu)

        dy = dy.reshape([-1, 1, J])
        mu = mu.reshape([-1, I])
        kl1 = kl_div(torch.sum(torch.mul(K, dy), -1), mu)
        assert not torch.isnan(kl1).any()
        dx = dx.reshape([-1, I])
        cons1 = torch.sum(torch.mul(kl1, dx), -1)
        assert not torch.isnan(cons1).any()

        dx = dx.reshape([-1, I, 1])
        nu = nu.reshape([-1, J])
        kl2 = kl_div(torch.sum(torch.mul(K, dx), -2), nu)
        assert not torch.isnan(kl2).any()
        dy = dy.reshape([-1, J])
        cons2 = torch.sum(torch.mul(kl2, dy), -1)
        assert not torch.isnan(cons2).any()\


        constrain = 4 * (delta ** 2) * (cons1 + cons2)  # Convention: ...
        assert not torch.isnan(constrain).any()

        # Convention: ...
        transport = 4 * (delta ** 2) * \
            (torch.sum(
                torch.sum(
                    torch.mul(
                        torch.mul(dx, torch.mul(K, C)),
                        dy),
                    -1),
                -1)
             )
        assert not torch.isnan(transport).any()

        # Convention: int |x - y|^2, some literature use int |x - y|^2 / 2.
        p_opt = constrain + transport
        ctx.save_for_backward(K, C, torch.tensor(delta))

        assert not torch.isnan(p_opt).any()

        return p_opt, transport, constrain, K

    @staticmethod
    def backward(ctx, grad_output, grad_transport, grad_constrain, grad_K):
        """
        backward propagation
        :param ctx: save the contex
        :param grad_output: the gradiant that used to have gradient
        :param grad_transport: placeholder, not use
        :param grad_constrain: placeholder, not use
        :param grad_K: placeholder, not use
        :return:
        """
        K, C, delta = ctx.saved_tensors
        device = K.device
        batchSize, I, J = K.shape
        dx = torch.tensor([1 / I] * I).reshape([1, I, 1]).to(device)
        dy = torch.tensor([1 / J] * J).reshape([1, 1, J]).to(device)
        grad_output = grad_output.detach().reshape([-1, 1, 1])
        grad_input = torch.mul(torch.mul(dx, K), dy) * \
            grad_output * 4 * delta ** 2
        return grad_input, None, None, None, None


class WFRPointCloudLoss(nn.Module):
    def __init__(self, epsilon, niter, device):
        super(WFRPointCloudLoss, self).__init__()
        self.epsilon = epsilon
        self.niter = niter
        self.device = device

    def forward(self, C):
        return WFRPointCloudCostFunc.apply(
            C, self.epsilon, self.niter, self.device)


def sinkhorn_iteration(C, mu, nu, epsilon, numIterMax=1000, tol=1e-9, debug=True):
    """
    Sinkhorn iteration for balanced Optimal Transport
    Parameters:
        C: [*, I, J] cost matrix, up to 3 dimension, the first for batch size, used in grouping or barycenter
        mu: [*, I, 1] source margin distribution
        nu: [*, 1, J] target margin distribution
        epsilon: entropy regulizer
        numIterMax: number of iteration
    Return:
        transport: the primal distance
        margin1: the KL divergence of the first marginal and source distribution
        margin2: the KL divergence of the second marginal and target distribution
        K: the transport plan
    """
    *_, I, J = C.shape
    _, I1, _ = mu.shape
    *_, J1 = nu.shape

    if debug:
        assert I == I1
        assert J == J1
        assert len(C.shape) == len(mu.shape)
        assert len(C.shape) == len(nu.shape)

    def K_calc(_u, _v):
        _K = myclamp(torch.exp((_u + _v - C) / epsilon))
        if debug:
            assert not torch.isnan(_K).any()
        return _K

    u = torch.zeros_like(mu)  # Kantorovich potential for source distribution

    b = torch.ones_like(nu)  # partial update of Kantorovich potential
    v = torch.zeros_like(nu)  # Kantorovich potential for target distribution
    K = K_calc(u, v)
    transport = torch.sum(torch.sum(torch.mul(K, C), -1), -1).squeeze()
    for ii in range(numIterMax):
        s = torch.sum(torch.mul(K, b), -1, keepdim=True)
        a = myclamp(torch.div(mu, s))

        s = torch.sum(torch.mul(K, a), -2, keepdim=True)
        b = myclamp(torch.div(nu, s))

        if ii % 10 == 0 or torch.max(a) > big or torch.max(b) > big or ii == numIterMax - 1:
            u += epsilon * torch.log(a)
            v += epsilon * torch.log(b)
            K = K_calc(u, v)
            b = torch.ones_like(nu)

            transport_new = torch.sum(torch.sum(torch.mul(K, C), -1), -1).squeeze()
            if abs(transport_new - transport) / abs(transport) < tol:
                break
            else:
                transport = transport_new

    kl1 = kl_div(torch.sum(K, -1, keepdim=True), mu)
    margin1 = torch.sum(kl1, -2).squeeze()
    kl2 = kl_div(torch.sum(K, -2, keepdim=True), nu)
    margin2 = torch.sum(kl2, -1).squeeze()

    return transport_new, margin1, margin2, K


def WFR_sinkhorn_iteration(C, mu, nu, epsilon, numIterMax=100, tol=1e-9, debug=True):
    """
    Sinkhorn iteration for balanced Optimal Transport
    Parameters:
        C: [*, I, J] cost matrix, up to 3 dimension, the first for batch size, used in grouping or barycenter
        mu: [*, I, 1] source margin distribution
        nu: [*, 1, J] target margin distribution
        epsilon: entropy regulizer
        numIterMax: number of iteration
    Return:
        transport: the primal distance
        margin1: the KL divergence of the first marginal and source distribution
        margin2: the KL divergence of the second marginal and target distribution
        K: the transport plan
    """
    *_, I, J = C.shape
    _, I1, _ = mu.shape
    *_, J1 = nu.shape

    if debug:
        assert I == I1
        assert J == J1
        assert len(C.shape) == len(mu.shape)
        assert len(C.shape) == len(nu.shape)

    def K_calc(_u, _v):
        _K = myclamp(torch.exp((_u + _v - C) / epsilon))
        if debug:
            assert not torch.isnan(_K).any()
        return _K

    u = torch.zeros_like(mu)  # Kantorovich potential for source distribution

    b = torch.ones_like(nu)  # partial update of Kantorovich potential
    v = torch.zeros_like(nu)  # Kantorovich potential for target distribution
    K = K_calc(u, v)
    transport = torch.sum(torch.sum(torch.mul(K, C), -1), -1).squeeze()
    for ii in range(numIterMax):
        s = torch.sum(torch.mul(K, b), -1, keepdim=True)
        a = myclamp(torch.div(mu, s) ** (1/(1+epsilon)))

        s = torch.sum(torch.mul(K, a), -2, keepdim=True)
        b = myclamp(torch.div(nu, s) ** (1/(1+epsilon)))

        if ii % 10 == 0 or torch.max(a) > big or torch.max(b) > big or ii == numIterMax - 1:
            u += epsilon * torch.log(a)
            v += epsilon * torch.log(b)
            K = K_calc(u, v)
            b = torch.ones_like(nu)

            transport_new = torch.sum(torch.sum(torch.mul(K, C), -1), -1).squeeze()
            if abs(transport_new - transport) / abs(transport) < tol:
                break
            else:
                transport = transport_new

    kl1 = kl_div(torch.sum(K, -1, keepdim=True), mu)
    margin1 = torch.sum(kl1, -2).squeeze()
    kl2 = kl_div(torch.sum(K, -2, keepdim=True), nu)
    margin2 = torch.sum(kl2, -1).squeeze()

    return transport_new, margin1, margin2, K


def gsinkhorn_iteration(C, mu, nu, lambdda, epsilon, numIterMax=100, tol=1e-6, debug=False):
    """
    Generalized Sinkhorn iteration for balanced Optimal Transport
    Parameters:
        C: [*, I, J] cost matrix, up to 3 dimension, the first for batch size, used in grouping or barycenter
        mu: [*, I, 1] source margin distribution
        nu: [*, 1, J] target margin distribution
        lambdda: the lambda coefficient for two KL relax term
        epsilon: entropy regulizer
        numIterMax: number of iteration
    Return:
        transport: the primal distance
        margin1: the KL divergence of the first marginal and source distribution
        margin2: the KL divergence of the second marginal and target distribution
        K: the transport plan
    """
    *_, I, J = C.shape
    _, I1, _ = mu.shape
    *_, J1 = nu.shape

    if debug:
        assert I == I1
        assert J == J1
        assert len(C.shape) == len(mu.shape)
        assert len(C.shape) == len(nu.shape)

    def K_calc(_u, _v):
        _K = myclamp(torch.exp((_u + _v - C) / epsilon))
        if debug:
            assert not torch.isnan(_K).any()
        return _K

    pow_coef = lambdda / (lambdda + epsilon)

    u = torch.zeros_like(mu)  # Kantorovich potential for source distribution

    b = torch.ones_like(nu)  # partial update of Kantorovich potential
    v = torch.zeros_like(nu)  # Kantorovich potential for target distribution
    K = K_calc(u, v)
    transport = torch.sum(torch.sum(torch.mul(K, C), -1), -1).squeeze()
    for ii in range(numIterMax):
        s = torch.sum(torch.mul(K, b), -1, keepdim=True)
        a = myclamp(torch.div(mu, s) ** pow_coef)

        s = torch.sum(torch.mul(K, a), -2, keepdim=True)
        b = myclamp(torch.div(nu, s) ** pow_coef)

        if ii % 10 == 0 or torch.max(a) > big or torch.max(b) > big or ii == numIterMax - 1:
            u += epsilon * torch.log(a)
            v += epsilon * torch.log(b)
            K = K_calc(u, v)
            b = torch.ones_like(nu)

            transport_new = torch.sum(torch.sum(torch.mul(K, C), -1), -1).squeeze()
            if abs(transport_new - transport) / abs(transport) < tol:
                # print('meet tol, jump out of sinkhorn iteration')
                break
            else:
                transport = transport_new

    kl1 = kl_div(torch.sum(K, -1, keepdim=True), mu)
    margin1 = torch.sum(kl1, -2).squeeze()
    kl2 = kl_div(torch.sum(K, -2, keepdim=True), nu)
    margin2 = torch.sum(kl2, -1).squeeze()

    return transport, margin1, margin2, K


def forward_relax_sinkhorn_iteration(C, mu, nu, lambdda, epsilon, numIterMax=100, tol=1e-6, debug=False):
    """
    Generalized Sinkhorn iteration for balanced Optimal Transport
    Parameters:
        C: [*, I, J] cost matrix, up to 3 dimension, the first for batch size, used in grouping or barycenter
        mu: [*, I, 1] source margin distribution
        nu: [*, 1, J] target margin distribution
        lambdda: the lambda coefficient for two KL relax term
        epsilon: entropy regulizer
        numIterMax: number of iteration
    Return:
        transport: the primal distance
        margin1: the KL divergence of the first marginal and source distribution
        margin2: the KL divergence of the second marginal and target distribution
        K: the transport plan
    """
    *_, I, J = C.shape
    _, I1, _ = mu.shape
    *_, J1 = nu.shape

    if debug:
        assert I == I1
        assert J == J1
        assert len(C.shape) == len(mu.shape)
        assert len(C.shape) == len(nu.shape)

    def K_calc(_u, _v):
        _K = myclamp(torch.exp((_u + _v - C) / epsilon))
        if debug:
            assert not torch.isnan(_K).any()
        return _K

    pow_coef = lambdda / (lambdda + epsilon)

    u = torch.zeros_like(mu)  # Kantorovich potential for source distribution

    b = torch.ones_like(nu)  # partial update of Kantorovich potential
    v = torch.zeros_like(nu)  # Kantorovich potential for target distribution
    K = K_calc(u, v)
    transport = torch.sum(torch.sum(torch.mul(K, C), -1), -1).squeeze()
    for ii in range(numIterMax):
        s = torch.sum(torch.mul(K, b), -1, keepdim=True)
        a = myclamp(torch.div(mu, s) ** pow_coef)

        s = torch.sum(torch.mul(K, a), -2, keepdim=True)
        b = myclamp(torch.div(nu, s) ** pow_coef)

        if ii % 10 == 0 or torch.max(a) > big or torch.max(b) > big or ii == numIterMax - 1:
            u += epsilon * torch.log(a)
            v += epsilon * torch.log(b)
            K = K_calc(u, v)
            b = torch.ones_like(nu)

            transport_new = torch.sum(torch.sum(torch.mul(K, C), -1), -1).squeeze()
            if abs(transport_new - transport) / abs(transport) < tol:
                # print('meet tol, jump out of sinkhorn iteration')
                break
            else:
                transport = transport_new

    kl1 = kl_div(torch.sum(K, -1, keepdim=True), mu)
    margin1 = torch.sum(kl1, -2).squeeze()
    kl2 = kl_div(torch.sum(K, -2, keepdim=True), nu)
    margin2 = torch.sum(kl2, -1).squeeze()

    return transport, margin1, margin2, K

# ====== TODO: RE-implement the following four class to provide the differentiable loss functions ========

class WPointCloudCostFunc(Function):
    @staticmethod
    def forward(ctx, C, epsilon, niter, device):
        C = C.detach().to(device)
        batch_size, I, J = C.shape
        mu = torch.tensor([1.0] * I).to(device)
        nu = torch.tensor([1.0] * J).to(device)
        dx = torch.tensor([1 / I] * I).to(device)
        dy = torch.tensor([1 / J] * J).to(device)

        # proxdiv_F(s, u, epsilon) <- epsilon (only once)
        def K_calc(_u, _v):
            return torch.clamp(
                torch.exp(
            (-C + _u.reshape([-1, I, 1]) + _v.reshape([-1, 1, J])) / epsilon), 
                                          0, huge)

        b = torch.ones_like(nu)
        # proxdiv_F(s, u, epsilon) <- u = u (initialization)
        u = torch.zeros_like(mu)
        # proxdiv_F(s, u, epsilon) <- u = v (initialization)
        v = torch.zeros_like(nu)
        K = K_calc(u, v)
        if torch.isnan(K).any():
            print(u.reshape([1, -1]), v.reshape([1, -1]))

        for ii in range(niter):
            b = b.reshape([-1, 1, J])
            dy = dy.reshape([-1, 1, J])
            bdy = torch.mul(b, dy)
            # proxdiv_F(s, u, epsilon) <- s = K * bdy (update)
            s = torch.sum(torch.mul(K, bdy), -1)
            # proxdiv_F(s, u, epsilon)
            a = torch.clamp(torch.div(mu, s), 0, huge)

            a = a.reshape([-1, I, 1])
            dx = dx.reshape([-1, I, 1])
            adx = torch.mul(a, dx)
            # proxdiv_F(s, u, epsilon) <- s = K * adx (update)
            s = torch.sum(torch.mul(K, adx), -2)

            # proxdiv_F(s, u, epsilon)
            b = torch.clamp(torch.div(nu, s), 0, huge)
            if torch.max(a) > big or torch.max(b) > big or ii == niter - 1:
                # proxdiv_F(s, u, epsilon) <- u = u (update)
                u = epsilon * torch.log(a).squeeze() + u
                # proxdiv_F(s, u, epsilon) <- u = v (update)
                v = epsilon * torch.log(b).squeeze() + v
                K = K_calc(u, v)
                if torch.isnan(K).any():
                    print(u.reshape([1, -1]), v.reshape([1, -1]))
                b = torch.ones_like(nu)

        dy = dy.reshape([-1, 1, J])
        mu = mu.reshape([-1, I])
        kl1 = kl_div(torch.sum(torch.mul(K, dy), -1), mu)
        assert not torch.isnan(kl1).any()
        dx = dx.reshape([-1, I])
        cons1 = torch.sum(torch.mul(kl1, dx), -1)
        assert not torch.isnan(cons1).any()

        dx = dx.reshape([-1, I, 1])
        nu = nu.reshape([-1, J])
        kl2 = kl_div(torch.sum(torch.mul(K, dx), -2), nu)
        assert not torch.isnan(kl2).any()
        dy = dy.reshape([-1, J])
        cons2 = torch.sum(torch.mul(kl2, dy), -1)
        assert not torch.isnan(cons2).any()

        constrain = cons1 + cons2  # torch.Tensor([0]) # Convention: ...
        assert not torch.isnan(constrain).any()

        # Convention: ...
        transport = torch.sum(
            torch.sum(torch.mul(torch.mul(dx, torch.mul(K, C)), dy), -1), -1)
        if torch.isinf(K).any():
            print(torch.isinf(K), "inf")
        if torch.isnan(K).any():
            print(torch.isnan(K), "nan")
        if torch.isinf(C).any():
            print(torch.isinf(C), "inf")
        if torch.isnan(C).any():
            print(torch.isnan(C), "nan")
        try:
            assert not torch.isnan(transport).any()
        except:
            import pickle
            with open("dump_4.pickle", mode='wb') as f:
                pickle.dump([
                    K.cpu(),
                    C.cpu(),
                    dx.cpu(),
                    dy.cpu(),
                    mu.cpu(),
                    nu.cpu(),
                    a.cpu(),
                    b.cpu(),
                    u.cpu(),
                    v.cpu()
                ], f)

        # Convention: int |x - y|^2, some literature use int |x - y|^2 / 2.
        p_opt = constrain + transport
        ctx.save_for_backward(K, C)

        assert not torch.isnan(p_opt).any()

        return p_opt, transport, constrain, K

    @staticmethod
    def backward(ctx, grad_output, grad_transport, grad_constrain, grad_K):
        K, C = ctx.saved_tensors
        device = K.device
        batchSize, I, J = K.shape
        dx = torch.tensor([1 / I] * I).reshape([1, I, 1]).to(device)
        dy = torch.tensor([1 / J] * J).reshape([1, 1, J]).to(device)
        grad_output = grad_output.detach().reshape([-1, 1, 1])
        grad_input = torch.mul(torch.mul(torch.mul(dx, K), dy), grad_output)
        # Only the gradient of KP.
        # print(grad_input.shape)
        try:
            assert not torch.isnan(grad_output).any()
        except:
            import pickle
            with open("backward_parameters.pickle", 'wb') as f:
                pickle.dump([
                    K.cpu(), C.cpu(), grad_output.cpu()
                ], f)
            raise KeyboardInterrupt
        return grad_input, None, None, None


class WPointCloudLoss(nn.Module):
    def __init__(self, epsilon, niter, device):
        super(WPointCloudLoss, self).__init__()
        self.epsilon = epsilon
        self.niter = niter
        self.device = device

    def forward(self, D):
        return WPointCloudCostFunc.apply(D, self.epsilon, self.niter, self.device)

# Reminder: For W2 sinkhorn, epsilon-decreasing is sometimes necessary. For instance,
# the sinkhorn between two Diracs with distance 4 can not begin with epsilon = 1e-3,
# It must begin with epsilon = 1e-2. (Since we are using float type, we should pay more
# attention to such pitfalls.)


class WFRFlexPointCloudCostFunc(Function):

    @staticmethod
    def forward(ctx, C, delta, epsilon, niter, mu=None, nu=None, device=default_device):
        C = C.to(device).detach()
        batchSize, I, J = C.shape
        nonemu = torch.tensor(mu is None)
        nonenu = torch.tensor(nu is None)
        if mu is None:
            mu = torch.tensor([1.0] * I).to(device)
        else:
            mu = mu.to(device).detach()
        if nu is None:
            nu = torch.tensor([1.0] * J).to(device)
        else:
            nu = nu.to(device).detach()

        dx = torch.tensor([1 / I] * I).reshape([1, I, 1]).to(device)
        dy = torch.tensor([1 / J] * J).reshape([1, 1, J]).to(device)
        p_coef = 1 / (1 + epsilon)

        def K_calc(_u, _v):
            return torch.clamp(torch.exp((-C + _u.reshape([-1, I, 1]) +
                                          _v.reshape([-1, 1, J])) / epsilon), 0, huge)
            # anonymous function that calculate the kernel K from cost C

        b = torch.ones_like(nu)
        # proxdiv_F(s, u, epsilon) <- u = u (initialization)
        u = torch.zeros_like(mu)
        # proxdiv_F(s, u, epsilon) <- u = v (initialization)
        v = torch.zeros_like(nu)
        K = K_calc(u, v)

        for ii in range(niter):
            b = b.reshape([-1, 1, J])
            dy = dy.reshape([-1, 1, J])
            bdy = torch.mul(b, dy)
            # proxdiv_F(s, u, epsilon) <- s = K * bdy (update)
            s = torch.sum(torch.mul(K, bdy), -1)
            # proxdiv_F(s, u, epsilon)
            a = torch.clamp(
                torch.pow(torch.div(mu, torch.exp(u) * s), p_coef), 0, huge)

            a = a.reshape([-1, I, 1])
            dx = dx.reshape([-1, I, 1])
            adx = torch.mul(a, dx)
            # proxdiv_F(s, u, epsilon) <- s = K * adx (update)
            s = torch.sum(torch.mul(K, adx), -2)

            # proxdiv_F(s, u, epsilon)
            b = torch.clamp(
                torch.pow(torch.div(nu, torch.exp(v) * s), p_coef), 0, huge)
            if torch.max(a) > big or torch.max(b) > big or ii == niter - 1:
                # proxdiv_F(s, u, epsilon) <- u = u (update)
                u = epsilon * torch.log(a).squeeze() + u
                # proxdiv_F(s, u, epsilon) <- u = v (update)
                v = epsilon * torch.log(b).squeeze() + v
                K = K_calc(u, v)
                b = torch.ones_like(nu)

        dy = dy.reshape([-1, 1, J])
        mu = mu.reshape([-1, I])
        rmu = torch.sum(torch.mul(K, dy), -1)
        kl1 = kl_div(rmu, mu)
        assert not torch.isnan(kl1).any()
        dx = dx.reshape([-1, I])
        cons1 = torch.sum(torch.mul(kl1, dx), -1)
        assert not torch.isnan(cons1).any()

        dx = dx.reshape([-1, I, 1])
        nu = nu.reshape([-1, J])
        rnu = torch.sum(torch.mul(K, dx), -2)
        kl2 = kl_div(rnu, nu)
        assert not torch.isnan(kl2).any()
        dy = dy.reshape([-1, J])
        cons2 = torch.sum(torch.mul(kl2, dy), -1)
        assert not torch.isnan(cons2).any()\


        constrain = cons1 + cons2
        assert not torch.isnan(constrain).any()

        transport = torch.sum(
            torch.sum(torch.mul(torch.mul(dx, torch.mul(K, C)), dy), -1), -1)
        assert not torch.isnan(transport).any()

        # Convention: int |x - y|^2, some literature use int |x - y|^2 / 2.
        p_opt = constrain + transport
        ctx.save_for_backward(K, rmu, rnu, mu, nonemu, nu,
                              nonenu, torch.tensor(delta))

        assert not torch.isnan(p_opt).any()

        return transport, cons1, cons2, K

    @staticmethod
    def backward(ctx, grad_transport, grad_cons1, grad_cons2, grad_K):
        K, rmu, rnu, mu, nonemu, nu, nonenu, delta = ctx.saved_tensors
        device = K.device
        batchSize, I, J = K.shape
        dx = torch.tensor([1 / I] * I).reshape([1, I, 1]).to(device)
        dy = torch.tensor([1 / J] * J).reshape([1, 1, J]).to(device)
        grad_transport = grad_transport.detach().reshape([-1, 1, 1])
        grad_C = torch.mul(torch.mul(dx, K), dy) * grad_transport
        grad_mu = (1 - rmu / mu) * dx if nonemu.item() == 0 else None
        grad_nu = (1 - rnu / nu) * dy if nonenu.item() == 0 else None
        return grad_C, None, None, None, grad_mu, grad_nu, None


class WFRFlexPointCloudLoss(nn.Module):
    def __init__(self, epsilon, niter, device=default_device):
        super(WFRFlexPointCloudLoss, self).__init__()
        self.epsilon = epsilon
        self.niter = niter
        self.device = device
        self.delta = 1

    def set_delta(self, new_delta):
        self.delta = new_delta
        print("set new delta", self.delta)

    def wfrcost_matrix(self, D, delta=None):
        if delta is None:
            return wfrcost_matrix(D, self.delta, device=self.device)
        else:
            return wfrcost_matrix(D, delta, device=self.device)

    def forward(self, C, mu=None, nu=None):
        return WFRFlexPointCloudCostFunc.apply(
            C, self.delta, self.epsilon, self.niter, mu, nu, self.device)



if __name__ == "__main__":
    X = torch.rand(100, 10)     
    Y = torch.rand(100, 10)
    # l2 ^ 2 distance matrix
    C = (X * X).sum(axis = 1, keepdim=True) + (Y * Y).sum(axis = 1) - 2 * (X.mm(Y.t())) 
    mu, nu = torch.rand(100), torch.rand(100)
    mu, nu = mu / torch.norm(mu), nu / torch.norm(nu)

    import ot
    P_numpy = ot.sinkhorn(mu.squeeze().numpy(), nu.squeeze().numpy(), C.numpy(), 1, method = 'greenkhorn')
    mu, nu, C = mu.unsqueeze(1).unsqueeze(0), nu.unsqueeze(0).unsqueeze(0), C.unsqueeze(0)
    _, _, _, P_torch = sinkhorn_iteration(C, mu, nu, 1) 

    print("error between P_numpy and P_torch: {}".format(np.linalg.norm(P_numpy - P_torch.numpy())))

