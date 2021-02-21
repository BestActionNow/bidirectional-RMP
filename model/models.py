import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import os
import numpy as np
import json
from collections import OrderedDict
from torch.autograd import Variable

from copy import deepcopy
from evaluation import CSLS
from data import WordDictionary, MonoDictionary, Language,\
    CrossLingualDictionary, GaussianAdditive
from utils import to_cuda, to_numpy
from SinkhornOT import gw_iterative_1, get_intra_sim, cos_dist_mat, norm_dist_mat 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TwoLayerTrans(nn.Module):
    def __init__(self, embed_dim, init = None):
        super(LinearTrans, self).__init__()
        hidden = 200
        self.transform1 = nn.Linear(embed_dim, hidden, bias=False)
        self.transform2 = nn.Linear(hidden, embed_dim, bias=False)

        if init != None:
            self.init_weights(init)
        self.init = init

    def forward(self, inp):
        output = self.transform1(inp)
        output = self.transform2(outpu)
        return output

class LinearTrans(nn.Module):
    def __init__(self, embed_dim, init = None):
        super(LinearTrans, self).__init__()
        self.transform = nn.Linear(embed_dim, embed_dim, bias=False)
        if init != None:
            self.init_weights(init)
        self.init = init

    def forward(self, inp):
        return self.transform(inp)
    
    def setWeight(self, Q):
        W = self.transform.weight.data
        W.copy_(Q.t())

    def _initialize(self):
        assert hasattr(self, 'init')
        self.init_weights(self.init)

    def init_weights(self, init):
        if init == 'ortho':
            nn.init.orthogonal(
                self.transform.weight, gain=nn.init.calculate_gain('linear'))
        elif init == 'eye':
            nn.init.eye_(self.transform.weight)
        else:
            raise NotImplementedError("{0} not supported".format(init))

    def orthogonalize(self, ortho_type="basic", beta=0.001):
        """
        Perform the orthogonalization step on the generated W matrix
        """
        if ortho_type == "basic":
            W = self.transform.weight.data
            o = ((1 + beta) * W) - (beta * W.mm(W.t().mm(W)))
            W.copy_(o)
        elif ortho_type == "spectral":
            self.spectral()
        elif ortho_type == "forbenius":
            self.forbenius()
        else:
            raise NotImplementedError(f"{ortho_type} not found")

    def spectral(self):
        W = self.transform.weight.data
        u, sigma, v = torch.svd(W)
        sigma_clamped = sigma.clamp(max=1.)
        new_W = torch.mm((torch.mm(u, torch.diag(sigma_clamped))), v.t())
        W.copy_(new_W)

    def forbenius(self):
        W = self.transform.weight.data
        fnorm = (W ** 2).sum() + 1e-6
        new_W = W / fnorm
        W.copy_(new_W)

class bliMethod(object):
    """
    the base class for bilingual lexicon induction methods
    """
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
        self.src = src
        self.tgt = tgt
        self.device = "cuda:{}".format(cuda)
        if seed > 0:
            self.set_seed(seed)
        self.batcher = batcher
        self.save_dir = save_dir
        self.evaluator = Evaluator(self.batcher[self.src], self.batcher[self.tgt], data_dir, save_dir)
        self.best_metrics = None
        self.best_eval_metric = -1.0
        self.best_model_state = None
        self.best_supervised_metric = -1.0
        # self.Q = None
        self.transform = None

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


    def get_checkpoint(self, metrics):
        """
            Sends back the checkpoint which stores relevant information about the model
            :param metrics (dict: str -> float): The dictionary of metrics
        """
        assert all(metric in metrics for metric in ["acc1", "acc5", "acc10", "unsupervised", "total"]), "Not all metrics found"
        checkpoint = OrderedDict()
        for metric in metrics:
            checkpoint[metric] = metrics[metric]
        checkpoint['map_params'] = self.transform.state_dict()
        return checkpoint

    def log(self, eval_list=["supervised", "unsupervised", "monolingual"], save=False, eval_metric='unsupervised', **kwargs):
        csls = self.get_csls()
        metrics = self.evaluator.evaluate(csls, eval_list)
        logger = logging.getLogger(__name__)
        if metrics[eval_metric] > self.best_eval_metric:
            new_metric = metrics[eval_metric]
            logger.info(f"Metric {eval_metric} improved from {self.best_eval_metric:2.2f} to {new_metric:2.2f}")
            self.best_eval_metric = metrics[eval_metric]
            self.best_metrics = {metric: np.float64(metrics[metric]) if metric != 'monolingual' else metrics[metric] for metric in metrics}
            self.best_supervised_metric = metrics['acc1']
            self.best_model_state = self.transform.state_dict()
            if save:
                # Save model
                checkpoint = self.get_checkpoint(metrics)
                model_file = 'checkpoint.pth.tar' if 'model_file' not in kwargs else kwargs['model_file']
                checkpoint_path = os.path.join(self.save_dir, 'best_model', model_file)
                torch.save(checkpoint, checkpoint_path)
                # Save metrics
                metrics_file = 'metrics.json' if 'metrics_file' not in kwargs else kwargs['metrics_file']
                metrics_path = os.path.join(self.save_dir, 'best_model', metrics_file)
                with open(metrics_path, 'w') as fp:
                    json.dump(self.best_metrics, fp)
        return metrics[eval_metric]


    def log1(self, eval_list=["supervised", "unsupervised", "monolingual"], save=False, eval_metric='unsupervised', **kwargs):
        csls = self.get_csls1()
        metrics = self.evaluator.evaluate(csls, eval_list)
        logger = logging.getLogger(__name__)
        if metrics[eval_metric] > self.best_eval_metric:
            new_metric = metrics[eval_metric]
            logger.info(f"Metric {eval_metric} improved from {self.best_eval_metric:2.2f} to {new_metric:2.2f}")
            self.best_eval_metric = metrics[eval_metric]
            self.best_metrics = {metric: np.float64(metrics[metric]) if metric != 'monolingual' else metrics[metric] for metric in metrics}
            self.best_supervised_metric = metrics['acc1']
            self.best_model_state = self.transform1.state_dict()
            if save:
                # Save model
                checkpoint = self.get_checkpoint(metrics)
                model_file = 'checkpoint.pth.tar' if 'model_file' not in kwargs else kwargs['model_file']
                checkpoint_path = os.path.join(self.save_dir, 'best_model', model_file)
                torch.save(checkpoint, checkpoint_path)
                # Save metrics
                metrics_file = 'metrics.json' if 'metrics_file' not in kwargs else kwargs['metrics_file']
                metrics_path = os.path.join(self.save_dir, 'best_model', metrics_file)
                with open(metrics_path, 'w') as fp:
                    json.dump(self.best_metrics, fp)
        return metrics[eval_metric]

    def log2(self, eval_list=["supervised", "unsupervised", "monolingual"], save=False, eval_metric='unsupervised', **kwargs):
        csls = self.get_csls2()
        metrics = self.evaluator.evaluate(csls, eval_list)
        logger = logging.getLogger(__name__)
        if metrics[eval_metric] > self.best_eval_metric:
            new_metric = metrics[eval_metric]
            logger.info(f"Metric {eval_metric} improved from {self.best_eval_metric:2.2f} to {new_metric:2.2f}")
            self.best_eval_metric = metrics[eval_metric]
            self.best_metrics = {metric: np.float64(metrics[metric]) if metric != 'monolingual' else metrics[metric] for metric in metrics}
            self.best_supervised_metric = metrics['acc1']
            self.best_model_state = self.transform2.state_dict()
            if save:
                # Save model
                checkpoint = self.get_checkpoint(metrics)
                model_file = 'checkpoint.pth.tar' if 'model_file' not in kwargs else kwargs['model_file']
                checkpoint_path = os.path.join(self.save_dir, 'best_model', model_file)
                torch.save(checkpoint, checkpoint_path)
                # Save metrics
                metrics_file = 'metrics.json' if 'metrics_file' not in kwargs else kwargs['metrics_file']
                metrics_path = os.path.join(self.save_dir, 'best_model', metrics_file)
                with open(metrics_path, 'w') as fp:
                    json.dump(self.best_metrics, fp)
        return metrics[eval_metric]

    def get_csls(self):
        source_word_embeddings = self.batcher.name2lang[self.src].embeddings
        dest_word_embeddings = self.batcher.name2lang[self.tgt].embeddings
        return CSLS(src=source_word_embeddings,
                    tgt=dest_word_embeddings,
                    map_src=self.transform,
                    gpu=self.device)


    def get_csls1(self):
        source_word_embeddings = self.batcher.name2lang[self.src].embeddings
        dest_word_embeddings = self.batcher.name2lang[self.tgt].embeddings
        return CSLS(src=source_word_embeddings,
                    tgt=dest_word_embeddings,
                    map_src=self.transform1,
                    gpu=self.device)

    def get_csls2(self):
        source_word_embeddings = self.batcher.name2lang[self.src].embeddings
        dest_word_embeddings = self.batcher.name2lang[self.tgt].embeddings
        return CSLS(src=source_word_embeddings,
                    tgt=dest_word_embeddings,
                    map_src=self.transform2,
                    gpu=self.device)

class Evaluator(object):
    """
        Computes different metrics for evaluation purposes.
        Currently supported evaluation metrics
            - Unsupervised CSLS metric for model selection
            - Cross-Lingual precision matches
    """

    def __init__(self, src_lang, tgt_lang, data_dir, save_dir):
        # Load Cross-Lingual Data
        """
            :param src_lang (Language): The source Language
            :param tgt_lang (Language): The Target Language
            :param data_dir (str): The data directory
                (assumes cross-lingual dictionaries are kept in
                data_dir/crosslingual/{src}-{tgt}.txt)
        """
        assert (isinstance(src_lang, Language) and
                isinstance(tgt_lang, Language) and
                isinstance(data_dir, str))
        self.data_dir = data_dir
        self.src_lang, self.tgt_lang = src_lang, tgt_lang
        self.logger = logging.getLogger()
        self.save_dir = save_dir

    def supervised(
        self, csls, metrics,
        mode="csls", nltk_flag=False, word_dict=None
    ):
        """
        Reports the precision at k (1, 5, 10) accuracy for
        word translation using facebook dictionaries
        """
        try:
            import nltk
            nltk.data.path = ['./nltk_data']
        except ImportError:
            pass
        if not hasattr(self, 'word_dict'):
            if self.data_dir == "./muse_data/":
                cross_lingual_dir = os.path.join(
                    self.data_dir, "crosslingual", "dictionaries")
                cross_lingual_file = "%s-%s.5000-6500.txt" % (
                    self.src_lang.name, self.tgt_lang.name)
                cross_lingual_file = os.path.join(
                    cross_lingual_dir, cross_lingual_file)
                self.word_dict = WordDictionary(
                    self.src_lang, self.tgt_lang, cross_lingual_file)
            elif self.data_dir == "./vecmap_data/":
                cross_lingual_dir = os.path.join(
                    self.data_dir, "dictionaries")
                cross_lingual_file = "%s-%s.test.txt" % (
                    self.src_lang.name, self.tgt_lang.name)
                cross_lingual_file = os.path.join(
                    cross_lingual_dir, cross_lingual_file)
                self.word_dict = WordDictionary(
                    self.src_lang, self.tgt_lang, cross_lingual_file)

        word_dict = word_dict or self.word_dict
        predictions, _ = self.get_match_samples(
            csls, word_dict.word_map[:, 0], 10, mode=mode, use_mean=False)
        _metrics, total = word_dict.precisionatk_nltk(
            predictions, [1, 5, 10]) \
            if nltk_flag else \
            word_dict.precisionatk(predictions, [1, 5, 10])
        metrics['total'] = total
        metrics['acc1'] = _metrics[0]
        metrics['acc5'] = _metrics[1]
        metrics['acc10'] = _metrics[2]
        total = metrics["total"]
        acc1 = metrics["acc1"]
        acc5 = metrics["acc5"]
        acc10 = metrics["acc10"]
        self.logger.info(
            f"Total: {total:d}, "
            f"Precision@1: {acc1:5.2f}, @5: {acc5:5.2f}, @10: {acc10:5.2f}"
        )
        return metrics

    def unsupervised(self, csls, metrics, mode="csls"):
        max_src_word_considered = 10000
        _, metrics['unsupervised'] = self.get_match_samples(
            csls, np.arange(
                min(self.src_lang.vocab, int(max_src_word_considered))),
            1, mode=mode)
        self.logger.info(
            "{0:12s}: {1:5.4f}".format(
                "Unsupervised", metrics['unsupervised']))
        return metrics

    def monolingual(self, csls, metrics, **kwargs):
        if not hasattr(self, 'mono_dict'):
            mono_lingual_dir = os.path.join(self.data_dir, "monolingual")
            self.mono_dict = MonoDictionary(self.src_lang, mono_lingual_dir)
        if not self.mono_dict.atleast_one:
            return metrics
        metrics = self.mono_dict.get_spearman_r(csls, metrics)
        self.logger.info("=" * 64)
        self.logger.info("{0:>25s}\t{1:5s}\t{2:10s}\t{3:5s}".format(
            "Dataset", "Found", "Not Found", "Corr"))
        self.logger.info("=" * 64)
        mean = -1
        for dname in metrics['monolingual']:
            if dname == "mean":
                mean = metrics['monolingual'][dname]
                continue
            found = metrics['monolingual'][dname]['found']
            not_found = metrics['monolingual'][dname]['not_found']
            correlation = metrics['monolingual'][dname]['correlation']
            self.logger.info(
                "{0:>25s}\t{1:5d}\t{2:10d}\t{3:5.4f}".format(
                    dname, found, not_found, correlation))
        self.logger.info("=" * 64)
        self.logger.info("Mean Correlation: {0:.4f}".format(mean))
        return metrics

    def crosslingual(self, csls, metrics, **kwargs):
        if not hasattr(self, 'cross_dict'):
            cross_lingual_dir = os.path.join(self.data_dir, "crosslingual")
            self.cross_dict = CrossLingualDictionary(
                self.src_lang, self.tgt_lang, cross_lingual_dir)
        if not self.cross_dict.atleast_one:
            return metrics
        metrics = self.cross_dict.get_spearman_r(csls, metrics)
        self.logger.info("=" * 64)
        self.logger.info(
            "{0:>25s}\t{1:5s}\t{2:10s}\t{3:5s}".format(
                "Dataset", "Found", "Not Found", "Corr"))
        self.logger.info("=" * 64)
        mean = -1
        for dname in metrics['crosslingual']:
            if dname == "mean":
                mean = metrics['crosslingual'][dname]
                continue
            found = metrics['crosslingual'][dname]['found']
            not_found = metrics['crosslingual'][dname]['not_found']
            correlation = metrics['crosslingual'][dname]['correlation']
            self.logger.info(
                "{0:>25s}\t{1:5d}\t{2:10d}\t{3:5.4f}".format(
                    dname, found, not_found, correlation))
        self.logger.info("=" * 64)
        self.logger.info("Mean Correlation: {0:.4f}".format(mean))

    def evaluate(self, csls, evals, mode="csls"):
        """
        Evaluates the csls object on the functions specified in evals
            :param csls: CSLS object, which contains methods to
                         map source space to target space and such
            :param evals: list(str): The functions to evaluate on
            :return metrics: dict: The metrics computed by evaluate.
        """
        metrics = {}
        for eval_func in evals:
            assert hasattr(self, eval_func), \
                "Eval Function {0} not found".format(eval_func)
            metrics = getattr(self, eval_func)(csls, metrics, mode=mode)
        return metrics

    def get_match_samples(self, csls, range_indices, n,
                          use_mean=True, mode="csls", hubness_thresh=20):
        """
        Computes the n nearest neighbors for range_indices (from src)
        wrt the target in csls object.
        For use_mean True, this computes the avg metric for all
        top k nbrs across batch, and reports the average top 1 metric.
            :param csls : The csls object
            :param range_indices: The source words (from 0, ... )
            :param n: Number of nbrs to find
            :param use_mean (bool): Compute the mean or not
        """
        target_indices, metric = csls.get_closest_csls_matches(
            source_indices=range_indices,
            n=n, mode=mode)
        if use_mean:
            logger = logging.getLogger(__name__)
            # Hubness removal
            unique_tgts, tgt_freq = np.unique(target_indices,
                                              return_counts=True)
            hubs = unique_tgts[tgt_freq > hubness_thresh]
            old_sz = metric.shape[0]
            filtered_metric = metric[np.isin(
                target_indices, hubs, invert=True)]
            logger.info(
                "Removed {0:d} hub elements For unsupervised".format(
                    old_sz - filtered_metric.shape[0]))
            mean_metric = np.mean(filtered_metric)
            return target_indices, mean_metric
        else:
            return target_indices, metric

    def hubness(self, csls, source_range=None):
        """
        Computes the distribution of the number of words source words of which a particular target word is a 
        nearest neighbour based on simple cosine similarity / CSLS metric
        """
        if source_range is None:
            source_range = self.src_lang.vocab
        logger = logging.getLogger()
        logger.info('Computing Hubness')
        fig = plt.figure()
        ax = plt.gca()

        for use_nn in [True, False]:
            target_indices, _ = csls.get_closest_csls_matches(np.arange(source_range), 1, False, use_nn = use_nn)
            _, freq = np.unique(target_indices, return_counts=True)
            freq = [w for w in freq if w > 20 and w < 80]
            ax.hist(freq, bins = 200, label = 'CSLS' if use_nn == False else 'NN')
        plt.xlabel('hubness count')
        plt.ylabel('frequency')
        plt.title('Hubness Measure for different metrics')
        ax.legend(loc = 'best')
        plt.savefig(os.path.join(self.save_dir, 'hubness_graph'))
        import sys; sys.exit()

class Hubness(object):
    def __init__(self, src_lang, tgt_lang, csls, save_dir, use_nn):
        assert isinstance(src_lang, Language) and isinstance(tgt_lang, Language)
        self.src_lang, self.tgt_lang = src_lang, tgt_lang
        self.save_dir = save_dir
        self.use_nn = use_nn
        self.csls = csls
        self.target_matches, _ = self.csls.get_closest_csls_matches(source_indices=np.arange(self.src_lang.vocab),
                                                                    n=1,
                                                                    use_mean=False,
                                                                    use_nn=use_nn)
        self.target_matches = self.target_matches[:, 0]

    def get_hubs_np(self, thresh):
        """
            :param thresh: The frequency threshold for hubs
            :return hubs: The index of target words that are hubs
        """
        unique_tgts, tgt_freq = np.unique(self.target_matches, return_counts=True)
        hubs = unique_tgts[tgt_freq > thresh]
        return hubs

    def get_hubs(self, thresh):
        '''
        returns a list of target words which are hubs
        '''
        frequency = {}
        for i in self.target_matches:
            if i not in frequency:
                frequency[i] = 0
            frequency[i] += 1
        hubs = [w for w in frequency.keys() if frequency[w] > thresh]
        return [self.tgt_lang.ix2word[w] for w in hubs]

    def get_neighbours(self, word):
        '''
        returns the source words which have the given target word as their nearest neighbour
        '''
        word = self.tgt_lang.word2ix[word]
        neighbours = [i for i in range(self.src_lang.vocab) if self.target_matches[i] == word]
        return [self.src_lang.ix2word[w] for w in neighbours]

    def get_hub_dict(self, thresh):
        '''
        Dumps a dictionary consisting of hubs and their neighbours
        '''
        hubs = self.get_hubs(thresh)
        word_dict = set()
        for i in hubs:
            neighbours = self.get_neighbours(i)
            for j in neighbours:
                word_dict.add((i, j))
        with open(os.path.join(self.save_dir, 'hub_dict'), 'w') as f:
            for (i, j) in word_dict:
                f.write("{0:s}\t{1:s}\n".format(i, j))
        return word_dict