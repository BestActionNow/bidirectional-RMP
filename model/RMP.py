import torch
import numpy as np
import time
import logging
import os
import json
import torch.optim as optim
from collections import OrderedDict

from sinkhorn import sinkhorn
from data import WordDictionary, MonoDictionary, Language,\
    CrossLingualDictionary, GaussianAdditive, Batcher
from torch.autograd import Variable
from SinkhornOT.sinkhorn_loss import forward_relax_sinkhorn_iteration
from evaluation import CSLS, Evaluator
from model import bliMethod, LinearTrans

class RMP(bliMethod):
    def __init__(self, src, tgt, cuda, seed, batcher, data_dir, save_dir):   
        """
        inputs:
            :param src (str): name of source lang
            :param tgt (str): name of target lang
            :param cuda (int): number of gpu to be used
            :param seed (int): seed for torch and numpy
            :param batcher (Batcher): The batcher object
            :param data_dir (str): Loaction of the data
            :param save_dir (str): Location to save models
        """
        super(RMP, self).__init__(src, tgt, cuda, seed, batcher, data_dir, save_dir)
        embed_dim = self.batcher.name2lang[self.src].embeddings.shape[1]
        self.transform = LinearTrans(embed_dim).double().to(self.device)
        self.Q = None

    def convex_init(self, X, Y, niter=100, reg=0.05, apply_sqrt=False):
        X, Y = X.cpu().numpy(), Y.cpu().numpy()
        import ot
        def sqrt_eig(x):
            U, s, VT = np.linalg.svd(x, full_matrices=False)
            return np.dot(U, np.dot(np.diag(np.sqrt(s)), VT))
        n, d = X.shape
        if apply_sqrt:
            X, Y = sqrt_eig(X), sqrt_eig(Y)
        K_X, K_Y = np.dot(X, X.T), np.dot(Y, Y.T)
        K_Y *= np.linalg.norm(K_X) / np.linalg.norm(K_Y)
        K2_X, K2_Y = np.dot(K_X, K_X), np.dot(K_Y, K_Y)
        P = np.ones([n, n]) / float(n)
        for it in range(1, niter + 1):
            G = np.dot(P, K2_X) + np.dot(K2_Y, P) - 2 * np.dot(K_Y, np.dot(P, K_X))
            q = ot.sinkhorn(np.ones(n), np.ones(n), G, reg, stopThr=1e-3, log=False)
            alpha = 2.0 / float(2.0 + it)
            P = alpha * q + (1.0 - alpha) * P
        obj = np.linalg.norm(np.dot(P, K_X) - np.dot(K_Y, P))
        U, s, V_t = np.linalg.svd(np.dot(Y.T, np.dot(P, X)))
        self.Q = torch.from_numpy(np.dot(U, V_t).T).to(self.device)
        self.transform.setWeight(self.Q)

    def P_solver(self, embi, embj, lambda_KL, epsilon):
        I, J = embi.shape[0], embj.shape[0]
        Mt = -torch.mm(embi.mm(self.Q), embj.t())
        mu = torch.ones(1, I, 1).to(torch.float32).to(self.device)
        nu = torch.ones(1, 1, J).to(torch.float32).to(self.device)
        Mt = Mt.view(1, I, J)
        _, _, _, P = forward_relax_sinkhorn_iteration(Mt, mu, nu, lambda_KL, epsilon)

        # Mt = -torch.mm(embi.mm(self.Q), embj.t())
        # ones = torch.ones(Mt.shape[0], device = self.device)
        # P, _ = sinkhorn(ones, ones, Mt, epsilon, stopThr = 1e-3)
        return P.squeeze()
    
    def binary_P(self, P):
        # convert P to binary matrix
        mx, _ = P.max(axis = 1, keepdim=True)
        binaP = torch.zeros_like(P, device= P.device)
        binaP[P >= mx] = 1
        return binaP

    def objective(self, n=5000):
        firstN = self.batcher.firstNbatch(n)
        Xn, Yn = firstN[self.src], firstN[self.tgt]
        Mt = -torch.mm(torch.mm(Xn, self.Q), Yn.t()).reshape(n, n)
        ones = torch.ones(Mt.shape[0], device=self.device)
        P, _ = sinkhorn(ones, ones, Mt, reg = 0.025, stopThr=1e-6)
        return 1000 * torch.norm(torch.mm(Xn, self.Q) -  torch.mm(P.squeeze(), Yn)) / n

    def orthogonal_mapping_update(self, GQ, learning_rate):
        next_Q = (self.Q - learning_rate * GQ).cpu().numpy()
        U, S, VT = np.linalg.svd(next_Q)
        self.Q = torch.from_numpy((U.dot(VT))).to(self.device)
        self.transform.setWeight(self.Q)
    
    def procrustes_onestep(self, src_aligned_embeddings, tgt_aligned_embeddings):
        matrix = torch.mm(src_aligned_embeddings.transpose(1, 0), tgt_aligned_embeddings)
        U, S, VT = np.linalg.svd(matrix.cpu().numpy())
        Q = torch.from_numpy((U.dot(VT))).to(self.device)
        return Q

    def supervised_rcsls_loss(self, src, tgt, nn_src, nn_tgt, k=10):
        # first an assert to ensure unit norming
        if not hasattr(self, "check_rcsls_valid"):
            self.check_rcsls_valid = True
            for l in self.batcher.name2lang.values():
                if l.unit_norm is False:
                    self.check_rcsls_valid = False
                    break
        if not self.check_rcsls_valid:
            raise RuntimeError("For RCSLS, need to unit norm")

        xtrans = self.transform(Variable(src))
        yvar = Variable(tgt)
        sup_loss = 2 * torch.sum(xtrans * yvar)
        # Compute nearest nn loss wrt src
        nn_tgt = Variable(nn_tgt)
        dmat = torch.mm(xtrans, nn_tgt.t())
        _, tix = torch.topk(dmat, k, dim=1)
        nnbrs = nn_tgt[tix.view(-1)].view((tix.shape[0], tix.shape[1], -1))
        nnbrs = Variable(nnbrs.data)  # Detach from compute graph
        nnloss = torch.bmm(nnbrs, xtrans.unsqueeze(-1)).squeeze(-1)
        nn_tgt_loss = torch.sum(nnloss) / k
        # Compute nearest nn loss wrt tgt
        nn_src = Variable(nn_src)
        nn_src_transform = Variable(self.transform(nn_src).data)
        dmat = torch.mm(yvar, nn_src_transform.t())
        _, tix = torch.topk(dmat, k, dim=1)
        nnbrs = nn_src[tix.view(-1)].view((tix.shape[0], tix.shape[1], -1))
        nnbrs = Variable(nnbrs.data)
        nnloss = torch.bmm(self.transform(nnbrs), yvar.unsqueeze(-1)).squeeze(-1)
        nn_src_loss = torch.sum(nnloss) / k
        return - (sup_loss - nn_tgt_loss - nn_src_loss) / src.size(0)

    def expand_dict(self, expand_dict_size=0., expand_tgt_rank=15000,
        expand_thresh=0., mode="csls", hubness_thresh=20):
        logger = logging.getLogger()
        src_emb = self.batcher.name2lang[self.src].embeddings
        tgt_emb = self.batcher.name2lang[self.tgt].embeddings
        # src_emb = src_emb[:expand_tgt_rank, ]
        # tgt_emb = tgt_emb[:expand_tgt_rank, ]
        src_emb = src_emb.mm(self.Q)
        assert src_emb.shape[0] == tgt_emb.shape[0]

        bs = 1024
        emb_num = src_emb.shape[0]
        # compute knn TODO:Reomve 
        src_knn = []
        tgt_knn = []
        for i in range(0, emb_num, bs):
            src_emb_slice = src_emb[i: min(i+bs, emb_num), ]
            sc = src_emb_slice.mm(tgt_emb.t())
            src_knn.append(torch.mean(sc.topk(10, 1)[0], 1).view(-1, 1))
        for i in range(0, emb_num, bs):
            tgt_emb_slice = tgt_emb[i: min(i+bs, emb_num), ]
            sc = tgt_emb_slice.mm(src_emb.t())
            tgt_knn.append(torch.mean(sc.topk(10, 1)[0], 1).view(1, -1))
        src_knn = torch.cat(src_knn, 0)
        tgt_knn = torch.cat(tgt_knn, 1)

        # src -> tgt
        tol_scores = []
        tol_targets = []
        for i in range(0, emb_num, bs):
            src_emb_slice = src_emb[i: min(i+bs, emb_num), ]
            sc = 2 * src_emb_slice.mm(tgt_emb.t()) - src_knn[i: min(i+bs, emb_num), ].view(-1, 1) - tgt_knn.view(1, -1)
            # sc = 2 * src_emb_slice.mm(tgt_emb.t())
            scores, targets = sc.topk(2, 1)
            tol_scores.append(scores)
            tol_targets.append(targets)
        tol_scores = torch.cat(tol_scores, 0)
        tol_targets = torch.cat(tol_targets, 0)
        pairs_s2t = torch.cat( 
            [torch.arange(tol_scores.shape[0]).unsqueeze(1).to(self.device), tol_targets[:, 0].unsqueeze(1)],
            1
        )

        diff1 = tol_scores[:, 0] - tol_scores[:, 1]
        reordered = diff1.sort(0, descending=True)[1]
        tol_scores = tol_scores[reordered]
        pairs_s2t = pairs_s2t[reordered]

        mask = (pairs_s2t.max(1)[0] <= expand_tgt_rank).unsqueeze(1)
        pairs_s2t = pairs_s2t.masked_select(mask).view(-1, 2)

        # tgt -> src
        tol_scores = []
        tol_targets = []
        for i in range(0, emb_num, bs):
            tgt_emb_slice = tgt_emb[i: min(i+bs, emb_num), ]
            sc = 2 * tgt_emb_slice.mm(src_emb.t()) - src_knn.view(1, -1) - tgt_knn[:,i: min(i+bs, emb_num) ].view(-1, 1)
            # sc = 2 * tgt_emb_slice.mm(src_emb.t())
            scores, targets = sc.topk(2, 1)
            tol_scores.append(scores)
            tol_targets.append(targets)
        tol_scores = torch.cat(tol_scores, 0)
        tol_targets = torch.cat(tol_targets, 0)
        pairs_t2s = torch.cat(
            [torch.arange(tol_scores.shape[0]).unsqueeze(1).to(self.device), tol_targets[:, 0].unsqueeze(1)],
            1
        )

        diff2 = tol_scores[:, 0] - tol_scores[:, 1]
        reordered = diff2.sort(0, descending=True)[1]
        tol_scores = tol_scores[reordered]
        pairs_t2s = pairs_t2s[reordered]

        mask = (pairs_t2s.max(1)[0] <= expand_tgt_rank).unsqueeze(1)
        pairs_t2s = pairs_t2s.masked_select(mask).view(-1, 2)
        pairs_t2s = torch.cat([pairs_t2s[:, 1].view(-1, 1), pairs_t2s[:, 0].view(-1, 1)], 1)
        

        # if expand_tgt_size >= 0:
        #     pairs_s2t = pairs_s2t[:expand_tgt_rank, ]
        #     pairs_t2s = pairs_t2s[:expand_tgt_rank, ]

        # combine 
        pairs_s2t = set([(a, b) for a, b in pairs_s2t.cpu().numpy()])
        pairs_t2s = set([(a, b) for a, b in pairs_t2s.cpu().numpy()])
        pairs = pairs_s2t & pairs_t2s
        pairs = torch.tensor([[a, b] for (a, b) in pairs]).cpu().numpy()

        if expand_dict_size >= 0:
            pairs = pairs[:expand_dict_size, ]
        return pairs


    def train(
        self, init_epsilon, init_iter, init_vocab, binary_P, lr, epoches, lambda_KL, epsilon, bsz, steps, refine_epochs, refine_dict_size, refine_tgt_rank, refine_thresh, num_tgts,
        sup_steps, sup_bsz, logafter, opt_params={"name": "SGD", "lr": 1.0}
        ):
        logger = logging.getLogger(__name__)
        logger.info("[W Proc. optimization]")
        # TODO: remove, get priority
        src, tgt, nn_src, nn_tgt = self.batcher.supervised_rcsls_minibatch(-1, self.src, self.tgt, 15000)
        weight = self.procrustes_onestep(src, tgt)
        Q_P = weight.t()
        # init by frank wolf
        firstN = self.batcher.firstNbatch(init_vocab)
        Xn, Yn = firstN[self.src], firstN[self.tgt]
        self.convex_init(Xn, Yn, init_iter, init_epsilon, apply_sqrt=True)
        # get csls for evaluation
        for epoch in range(epoches):
            logger.info("start of Epoch {}/{}".format(epoch+1, epoches))
            start = time.time()
            for it in range(1, steps + 1):
                # torch.cuda.empty_cache()
                # sample mini-batch
                mini_batch = self.batcher.minibatch(bsz)
                embi = mini_batch[self.src][1]
                embj = mini_batch[self.tgt][1]
                # PQ solver
                P = self.P_solver(embi, embj, lambda_KL, epsilon)
                if binary_P:
                    P = self.binary_P(P) 
                GQ = - torch.mm(embi.t(), P.mm(embj))
                self.orthogonal_mapping_update(GQ, lr/bsz)
                if it % 100 == 0:
                    print("    {}/{} iteration completes".format(it, steps), end="\r")
            steps, bsz = steps // 4, bsz * 2
            self.log()
            logger.info("Finished epoch ({0:d} / {1:d}). Took {2:.2f}s. Obj {3:.3f}.".format(epoch + 1, epoches, time.time() - start, self.objective()))

        # Refine Procedure
        sup_lr = opt_params["lr"]
        name = opt_params.pop("name")
        for epoch in range(refine_epochs):
            logger.info("start of refine Epoch {}/{}".format(epoch+1, refine_epochs))
            start = time.time()
            pairs = self.expand_dict(refine_dict_size, refine_tgt_rank, refine_thresh)
            self.batcher.update_supervised(self.src, self.tgt, pairs)
            # start optimization with RCSLS
            fold = np.inf
            opt_params["lr"] = sup_lr
            rcsls_optimizer = getattr(optim, name)(self.transform.parameters(), **opt_params)
            for iter in range(1, sup_steps+1):
                if opt_params["lr"] < 1e-4:
                    break
                rcsls_optimizer.zero_grad()
                src, tgt, nn_src, nn_tgt = self.batcher.supervised_rcsls_minibatch(sup_bsz, self.src, self.tgt, num_tgts)
                loss = self.supervised_rcsls_loss(
                    src, tgt, nn_src, nn_tgt)
                f = loss.item()
                lr_str = opt_params["lr"]
                if f > fold:
                    opt_params["lr"] /= 2
                    rcsls_optimizer = getattr(optim, name)(
                        self.transform.parameters(), **opt_params)
                    f = fold
                else:
                    loss.backward()
                    rcsls_optimizer.step()
                    self.Q = self.transform.transform.weight.data.t()
                if iter % logafter == 0:
                    print("RCSLS {}/{} iteration completes".format(iter, sup_steps), end="\r")
            self.log()
            logger.info("Finished refine epoch ({0:d} / {1:d}). Took {2:.2f}s. Obj {3:.3f}.".format(epoch + 1, refine_epochs, time.time() - start, self.objective()))
        
        logger.info("Finished Training after {0} epochs".format(epoches))
        logger.info("{0:12s}: {1:5.4f}".format("Unsupervised", self.best_metrics['unsupervised']))
        logger.info("Found {0:d} words for supervised metric. Precision@1: {1:5.2f}\t@5: {2:5.2f}\t@10: {3:5.2f}".format(int(self.best_metrics['total']),
            self.best_metrics['acc1'], self.best_metrics['acc5'], self.best_metrics['acc10']))
        