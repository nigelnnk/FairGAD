# -*- coding: utf-8 -*-
"""Outlier Aware Network Embedding for Attributed Networks (ONE)
"""
# Author: Xiyang Hu <xiyanghu@cmu.edu>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import time
import warnings

import numpy as np
import torch
from torch_geometric.utils import to_dense_adj
from scipy.stats import rankdata

from . import BaseDetector
from ..utils import validate_device, calculate_sp_loss, calculate_approx_ndcg_loss, hin_sp_loss, cor_sp_loss


class ONE_NEW(BaseDetector):
    """
    Outlier Aware Network Embedding for Attributed Networks

    .. note::
        This detector is transductive only. Using ``predict`` with
        unseen data will train the detector from scratch.

    See :cite:`bandyopadhyay2019outlier` for details.

    Parameters
    ----------
    hid_a : int, optional
        Hidden dimension for the attribute. Default: ``36``.
    hid_s : int, optional
        Hidden dimension for the structure. Default: ``36``.
    alpha : float, optional
        Weight for the attribute loss. Default: ``1.``.
    beta : float, optional
        Weight for the structural loss. Default: ``1.``.
    gamma : float, optional
        Weight for the combined loss. Default: ``1.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``5``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``-1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    """

    def __init__(self,
                 hid_a=36,
                 hid_s=36,
                 alpha=1.,
                 beta=1.,
                 gamma=1.,
                 weight_decay=0.,
                 contamination=0.1,
                 lr=0.004,
                 epoch=5,
                 gpu=-1,
                 verbose=0):
        super(ONE_NEW, self).__init__(contamination=contamination)

        self.hid_a = hid_a
        self.hid_s = hid_s
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.weight_decay = weight_decay
        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)
        self.verbose = verbose

        self.attribute_score_ = None
        self.structural_score_ = None
        self.combined_score_ = None

        self.model = None

    def fit(self, data, label=None):

        x, s = self.process_graph(data)

        num_nodes, in_dim = x.shape

        w = torch.randn(self.hid_a, self.hid_s).half().to(self.device)

        u = torch.randn(num_nodes, self.hid_a).half().to(self.device)
        v = torch.randn(self.hid_a, in_dim).half().to(self.device)

        g = torch.randn(num_nodes, self.hid_s).half().to(self.device)
        h = torch.randn(self.hid_s, num_nodes).half().to(self.device)

        self.model = ONEBase(g, h, u, v, w, self.alpha, self.beta, self.gamma).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        for epoch in range(self.epoch):
            x_, s_, diff = self.model()
            loss, o1, o2, o3 = self.model.loss_func(x,
                                                    x_,
                                                    s,
                                                    s_,
                                                    diff)

            self.attribute_score_ = o1.detach().cpu()
            self.structural_score_ = o2.detach().cpu()
            self.combined_score_ = o3.detach().cpu()
            self.decision_scores_ = ((o1 + o2 + o3) / 3).detach().cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._process_decision_scores()
        return self

    def fit_with_fairness(self, data, y_true=None, fair_factor=0.1, sens_var_col=None, adcg_factor=0.01,
                          regulariser="hin"):

        if adcg_factor > 0:
            self.fit(data)
            base_scores = self.decision_function(data)
            self.log_scores = False
            if base_scores.max() > 1000 or np.log10(np.exp2(base_scores).sum()) > 90:
                self.log_scores = True
                base_scores = np.log(base_scores)
            idcg_0 = np.sum((np.power(2, base_scores[sens_var_col == 0]) - 1) /
                             np.log2(rankdata(-base_scores[sens_var_col == 0]) + 1))
            idcg_1 = np.sum((np.power(2, base_scores[sens_var_col == 1]) - 1) /
                             np.log2(rankdata(-base_scores[sens_var_col == 1]) + 1))
            base_scores = torch.tensor(base_scores).to(self.device)

        x, s = self.process_graph(data)

        num_nodes, in_dim = x.shape

        w = torch.randn(self.hid_a, self.hid_s).to(self.device)

        u = torch.randn(num_nodes, self.hid_a).to(self.device)
        v = torch.randn(self.hid_a, in_dim).to(self.device)

        g = torch.randn(num_nodes, self.hid_s).to(self.device)
        h = torch.randn(self.hid_s, num_nodes).to(self.device)

        self.model = ONEBase(g, h, u, v, w, self.alpha, self.beta, self.gamma).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        for epoch in range(self.epoch):
            x_, s_, diff = self.model()
            loss, o1, o2, o3 = self.model.loss_func(x,
                                                    x_,
                                                    s,
                                                    s_,
                                                    diff)

            score = (o1 + o2 + o3) / 3
            self.attribute_score_ = o1.detach().cpu()
            self.structural_score_ = o2.detach().cpu()
            self.combined_score_ = o3.detach().cpu()
            self.decision_scores_ = score.detach().cpu().numpy()

            if sens_var_col is not None:
                curr_sensitive = sens_var_col.to(self.device)
                if regulariser == "hin":
                    sp_loss = hin_sp_loss(score, curr_sensitive, self.contamination)
                elif regulariser == "fairod":
                    sp_loss = calculate_sp_loss(score, curr_sensitive)
                elif regulariser == "correlation":
                    sp_loss = cor_sp_loss(score, curr_sensitive)

                approx_ndcg_loss = 0
                if adcg_factor > 0:
                    approx_ndcg_loss = calculate_approx_ndcg_loss(torch.log(score) if self.log_scores else score,
                                                                curr_sensitive,
                                                                base_scores,
                                                                idcg_0,
                                                                idcg_1)
                                                                
                if self.verbose:
                    print(f"Loss: {loss}\nSP Loss: {sp_loss}\nADCG Loss: {approx_ndcg_loss}")
                loss = loss + fair_factor * sp_loss + adcg_factor * approx_ndcg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._process_decision_scores()
        return self
    

    def decision_function(self, data, label=None):
        # if data is not None:
        #     warnings.warn("This detector is transductive only. "
        #                   "Training from scratch with the input data.")
        #     self.fit(data, label)
        outlier_scores = self.decision_scores_

        return outlier_scores

    def process_graph(self, data):
        x = data.x.to(self.device)
        s = to_dense_adj(data.edge_index)[0].to(self.device)
        return x, s


class ONEBase(torch.nn.Module):
    def __init__(self, g, h, u, v, w, alpha=1., beta=1., gamma=1.):

        super(ONEBase, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.g = torch.nn.Parameter(g)
        self.h = torch.nn.Parameter(h)
        self.u = torch.nn.Parameter(u)
        self.v = torch.nn.Parameter(v)
        self.w = torch.nn.Parameter(w)

    def forward(self):
        x_ = self.u @ self.v
        s_ = self.g @ self.h
        diff = self.g - self.u @ self.w
        return x_, s_, diff

    def loss_func(self, x, x_, s, s_, diff):
        dx = torch.sum(torch.pow(x - x_, 2), 1)
        o1 = dx / torch.sum(dx)
        loss_a = torch.mean(torch.log(torch.pow(o1, -1)) * dx)

        ds = torch.sum(torch.pow((s - s_).half(), 2), 1)
        o2 = ds / torch.sum(ds)
        loss_s = torch.mean(torch.log(torch.pow(o2, -1)) * ds)

        dc = torch.sum(torch.pow(diff, 2), 1)
        o3 = dc / torch.sum(dc)
        loss_c = torch.mean(torch.log(torch.pow(o3, -1)) * dc)

        loss = self.alpha * loss_a + self.beta * loss_s + self.gamma * loss_c

        return loss, o1, o2, o3
