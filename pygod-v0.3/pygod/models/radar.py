# -*- coding: utf-8 -*-
""" Residual Analysis for Anomaly Detection in Attributed Networks
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import warnings
from torch import nn
from pygod.metrics import *
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from sklearn.utils.validation import check_is_fitted
from scipy.stats import rankdata

from . import BaseDetector
from ..utils import validate_device, calculate_sp_loss, calculate_approx_ndcg_loss, hin_sp_loss, cor_sp_loss
from ..metrics import eval_roc_auc


class Radar(BaseDetector):
    """
    Radar (Residual Analysis for Anomaly Detection in Attributed
    Networks) is an anomaly detector with residual analysis. This
    model is transductive only.

    See :cite:`li2017radar` for details.

    Parameters
    ----------
    gamma : float, optional
        Loss balance weight for attribute and structure.
        Default: ``1.``.
    weight_decay : float, optional
        Weight decay (alpha and beta in the original paper).
        Default: ``0.01``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``5``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.models import Radar
    >>> model = Radar()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(None)
    """

    def __init__(self,
                 gamma=1.,
                 weight_decay=0.01,
                 lr=0.004,
                 epoch=100,
                 gpu=0,
                 contamination=0.1,
                 verbose=False):
        super(Radar, self).__init__(contamination=contamination)

        # model param
        self.gamma = gamma
        self.weight_decay = weight_decay

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)

        # other param
        self.verbose = verbose
        self.model = None

    def fit(self, G, y_true=None):
        """
        Fit detector with input data.

        Parameters
        ----------
        G : torch_geometric.data.Data
            The input data.
        y_true : numpy.ndarray, optional
            The optional outlier ground truth labels used to monitor
            the training progress. They are not used to optimize the
            unsupervised model. Default: ``None``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        G.s = to_dense_adj(G.edge_index)[0]
        x, s, l, w_init, r_init = self.process_graph(G)

        self.model = Radar_Base(w_init, r_init)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        for epoch in range(self.epoch):
            x_, r = self.model(x)
            loss = self._loss(x, x_, r, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            decision_scores = torch.sum(torch.pow(r, 2), dim=1).detach() \
                .cpu().numpy()

            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(epoch, loss.item()), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, decision_scores)
                    print(" | AUC {:.4f}".format(auc), end='')
                print()

        self.decision_scores_ = decision_scores
        self._process_decision_scores()
        return self

    def fit_with_fairness(self, G, y_true=None, fair_factor=0.1, sens_var_col=None, adcg_factor=0.01,
                          regulariser="hin"):
        """
        Fit detector with input data.

        Parameters
        ----------
        G : torch_geometric.data.Data
            The input data.
        y_true : numpy.ndarray, optional
            The optional outlier ground truth labels used to monitor
            the training progress. They are not used to optimize the
            unsupervised model. Default: ``None``.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        if adcg_factor > 0:
            self.fit(G)
            base_scores = self.decision_function(G)
            self.log_scores = False
            if base_scores.max() > 1000 or np.log10(np.exp2(base_scores).sum()) > 90:
                self.log_scores = True
                base_scores = np.log(base_scores)
            idcg_0 = np.sum((np.power(2, base_scores[sens_var_col == 0]) - 1) /
                             np.log2(rankdata(-base_scores[sens_var_col == 0]) + 1))
            idcg_1 = np.sum((np.power(2, base_scores[sens_var_col == 1]) - 1) /
                             np.log2(rankdata(-base_scores[sens_var_col == 1]) + 1))
            base_scores = torch.tensor(base_scores).to(self.device)

        G.s = to_dense_adj(G.edge_index)[0]
        x, s, l, w_init, r_init = self.process_graph(G)

        self.model = Radar_Base(w_init, r_init)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        for epoch in range(self.epoch):
            x_, r = self.model(x)
            loss = self._loss(x, x_, r, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            score = torch.sum(torch.pow(r, 2), dim=1)
            decision_scores = score.detach().cpu().numpy()

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

            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(epoch, loss.item()), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, decision_scores)
                    print(" | AUC {:.4f}".format(auc), end='')
                print()

        self.decision_scores_ = decision_scores
        self._process_decision_scores()
        return self

    def decision_function(self, G):
        """
        Predict raw anomaly score using the fitted detector. Outliers
        are assigned with larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        outlier_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        check_is_fitted(self, ['model'])

        # if G is not None:
        #     warnings.warn('The model is transductive only. '
        #                   'Training data is used to predict')

        outlier_scores = self.decision_scores_

        return outlier_scores

    def process_graph(self, G):
        """
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        x : torch.Tensor
            Attribute (feature) of nodes.
        """
        x = G.x.to(self.device)
        s = G.s.to(self.device)

        s = torch.max(s, s.T)
        l = self._comp_laplacian(s)

        w_init = torch.eye(x.shape[0], dtype=torch.half).to(self.device)
        temp = torch.eye(x.shape[0], dtype=torch.half).to(self.device)
        r_init = torch.inverse((1 + self.weight_decay) *
            temp + self.gamma * l) @ x
        del temp

        return x, s, l, w_init, r_init

    def _loss(self, x, x_, r, l):
        return torch.norm(x - x_ - r, 2) + \
               self.gamma * torch.trace(r.T @ l @ r)

    def _comp_laplacian(self, adj):
        d = torch.diag(torch.sum(adj, dim=1)).half()
        return d - adj


class Radar_Base(nn.Module):
    def __init__(self, w, r):
        super(Radar_Base, self).__init__()
        self.w = nn.Parameter(w)
        self.r = nn.Parameter(r)

    def forward(self, x):
        return self.w @ x, self.r
