from __future__ import absolute_import
import faiss
import numpy as np
from torch.autograd import Variable
import torch
import logging
import os

from data import Language, WordDictionary, MonoDictionary, CrossLingualDictionary
from utils import to_numpy, to_cuda

logger = logging.getLogger(__name__)

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
            cross_lingual_dir = os.path.join(
                self.data_dir, "crosslingual", "dictionaries")
            cross_lingual_file = "%s-%s.5000-6500.txt" % (
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
        import matplotlib.pyplot as plt
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
    

class CSLS(object):
    """
    Class that handles tasks related to Cross-domain Similarity Local Scaling
    """

    def __init__(self, src, tgt, map_src=None, map_tgt=None, k=10, gpu=None, gpu_device=0):
        """
        inputs:
            :param src (torch.tensor) : the source np.ndarray object
            :param tgt (torch.tensor) : the target np.ndarray object
            :param map_src (linear layer) : the Linear Layer for mapping the source (if applicable)
            :param map_tgt (linear layer) : the Linear Layer for mapping the target (if applicable)
            :param k (int) : the number of nearest neighbours to use (default: 10, as in paper)
        """
        if map_src is None:
            self.src = to_numpy(normalize(src), gpu)
        else:
            self.src = to_numpy(normalize(map_src(src)), gpu)

        if map_tgt is None:
            self.tgt = to_numpy(normalize(tgt), gpu)
        else:
            self.tgt = to_numpy(normalize(map_tgt(tgt)), gpu)

        self.k = k
        self.gpu = (gpu != None)
        self.gpu_device = gpu

        self.r_src = get_mean_similarity(self.src, self.tgt, self.k, self.gpu, self.gpu_device)
        self.r_tgt = get_mean_similarity(self.tgt, self.src, self.k, self.gpu, self.gpu_device)

    def map_to_tgt(self, source_indices):
        return self.src[source_indices, ...]

    def get_closest_csls_matches(self, source_indices, n, mode="csls"):
        """
        Gets the n closest matches of the elements located at the source indices in the target embedding.
        Returns: indices of closest matches and the mean CSLS of all these matches.
            This function maps the indices internally.
        inputs:
            :param source_indices (np.ndarray) : the source indices (in the source domain)
            :param n (int) : the number of closest matches to obtain
        """
        logger.info("Using Mode: {0}".format(mode))
        tgt_tensor = to_cuda(torch.Tensor(self.tgt), int(self.gpu_device[-1])).t()
        src_tensor = torch.Tensor(self.map_to_tgt(source_indices))

        r_src_tensor = to_cuda(torch.Tensor(self.r_src[source_indices, np.newaxis]), int(self.gpu_device[-1]))
        r_tgt_tensor = to_cuda(torch.Tensor(self.r_tgt[np.newaxis, ...]), int(self.gpu_device[-1]))

        batched_list = []
        batched_list_idx = []
        batch_size = 512
        for i in range(0, src_tensor.shape[0], batch_size):
            src_tensor_indexed = to_cuda(src_tensor[i: i + batch_size], int(self.gpu_device[-1]))
            r_src_tensor_indexed = r_src_tensor[i: i + batch_size]
            if mode == "nn":
                batch_scores = src_tensor_indexed.mm(tgt_tensor)
            elif mode == "csls":
                batch_scores = (2 * src_tensor_indexed.mm(tgt_tensor)) - r_src_tensor_indexed - r_tgt_tensor
            elif mode == "cdm":
                mu_x = torch.sqrt(1. - r_src_tensor_indexed)
                mu_y = torch.sqrt(1. - r_tgt_tensor)
                dxy = 1. - src_tensor_indexed.mm(tgt_tensor)
                eps = 1e-3
                batch_scores = -dxy / (mu_x + mu_y + eps)
            else:
                raise NotImplementedError("{0} not implemented yet".format(mode))
            best_scores, best_ix = batch_scores.topk(n)
            batched_list.append(best_scores)
            batched_list_idx.append(best_ix)
        return to_numpy(torch.cat(batched_list_idx, 0), self.gpu_device), to_numpy(torch.cat(batched_list, 0), self.gpu_device)


def get_faiss_nearest_neighbours(emb_src, emb_wrt, k, use_gpu=True, gpu_device=0):
    """
    Gets source points'/embeddings' nearest neighbours with respect to a set of target embeddings.
    inputs:
        :param emb_src (np.ndarray) : the source embedding matrix
        :param emb_wrt (np.ndarray) : the embedding matrix in which nearest neighbours are to be found
        :param k (int) : the number of nearest neightbours to find
        :param use_gpu (bool) : true if the gpu is to be used
        :param gpu_device (int) : the GPU to be used
    outputs:
        :returns distance (np.ndarray) : [len(emb_src), k] matrix of distance of
            each source point to each of its k nearest neighbours
        :returns indices (np.ndarray) : [len(emb_src), k] matrix of indices of
            each source point to each of its k nearest neighbours
    """
    if use_gpu:
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = int(gpu_device[-1])
        index = faiss.GpuIndexFlatIP(res, emb_wrt.shape[1], cfg)
    else:
        index = faiss.IndexFlatIP(emb_wrt.shape[1])
    index.add(emb_wrt.astype('float32'))
    return index.search(emb_src.astype('float32'), k)


def get_mean_similarity(emb_src, emb_wrt, k, use_gpu=True, gpu_device=0):
    """
    Gets the mean similarity of source embeddings with respect to a set of target embeddings.
    inputs:
        :param emb_src (np.ndarray) : the source embedding matrix
        :param emb_wrt (np.ndarray) : the embedding matrix wrt which the similarity is to be calculated
        :param k (int) : the number of points to be used to find mean similarity
        :param use_gpu (bool) : true if the gpu is to be used
        :param gpu_device (int) : the GPU to be used
    """
    nn_dists, _ = get_faiss_nearest_neighbours(emb_src, emb_wrt, k, use_gpu, gpu_device)
    return nn_dists.mean(1)


def normalize(arr):
    """
    Normalizes a vector of vectors into a vector of unit vectors
    """
    return arr / torch.norm(arr, p=2, dim=1).unsqueeze(1)


def _csls_test(range_indices):
    lang = Language('en')
    lang.load('wiki.en.test.vec')
    source_word_embeddings = lang.embeddings
    dest_word_embeddings = lang.embeddings

    csls = CSLS(source_word_embeddings, dest_word_embeddings, gpu=True)
    target_indices, mean_metric = csls.get_closest_csls_matches(range_indices, 1)

    print(target_indices)
    assert (target_indices == range_indices).all()


if __name__ == "__main__":
    _csls_test(range(20, 30))
