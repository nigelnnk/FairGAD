# -*- coding: utf-8 -*-
"""
Metrics used to evaluate the anomaly detection performance
"""
# Author: Yingtong Dou <ytongdou@gmail.com>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import warnings
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score


def eval_roc_auc(labels, pred):
    """
    ROC-AUC score for binary classification.

    Parameters
    ----------
    labels : numpy.ndarray
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : numpy.ndarray
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    roc_auc : float
        Average ROC-AUC score across different labels.
    """

    # outlier detection is a binary classification problem
    roc_auc = roc_auc_score(y_true=labels, y_score=pred)
    return roc_auc


def eval_recall_at_k(labels, pred, k):
    """
    Recall score for top k instances with the highest outlier scores.

    Parameters
    ----------
    labels : numpy.ndarray
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : numpy.ndarray
        Outlier scores in shape of ``(N, )``.
    k : int
        The number of instances to evaluate.

    Returns
    -------
    recall_at_k : float
        Recall for top k instances with the highest outlier scores.
    """

    N = len(pred)
    labels = np.array(labels)
    pred = np.array(pred)
    recall_at_k = sum(labels[pred.argpartition(N - k)[-k:]]) / sum(labels)

    return recall_at_k


def eval_precision_at_k(labels, pred, k):
    """
    Precision score for top k instances with the highest outlier scores.

    Parameters
    ----------
    labels : numpy.ndarray
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : numpy.ndarray
        Outlier scores in shape of ``(N, )``.
    k : int
        The number of instances to evaluate.

    Returns
    -------
    precision_at_k : float
        Precision for top k instances with the highest outlier scores.
    """

    N = len(pred)
    labels = np.array(labels)
    pred = np.array(pred)
    precision_at_k = sum(labels[pred.argpartition(N - k)[-k:]]) / k

    return precision_at_k


def eval_average_precision(labels, pred):
    """
    Average precision score for binary classification.

    Parameters
    ----------
    labels : numpy.ndarray
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : numpy.ndarray
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    ap : float
        Average precision score.
    """

    # outlier detection is a binary classification problem
    ap = average_precision_score(y_true=labels, y_score=pred)
    return ap


def eval_ndcg(labels, pred):
    """
    Normalized discounted cumulative gain for ranking.

    Parameters
    ----------
    labels : numpy.ndarray
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : numpy.ndarray
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    ndcg : float
        NDCG score.
    """

    labels = np.array(labels)
    pred = np.array(pred)
    if labels.dtype == bool:
        labels = labels.astype(int)
    ndcg = ndcg_score(y_true=[labels], y_score=[pred])
    return ndcg

def statistical_parity(pred, sensitive_var_dict):
    """
    Statistical parity of model prediction across different sensitive attribute values

    Parameters
    ----------
    pred : numpy.ndarray
        BINARY Outlier predictions (0 or 1) in shape of ``(N, )``where 1 represents outliers,
        0 represents normal instances.

    sensitive_var_dict: dictionary of key -> numpy.ndarray
        For each value of the sensitive attribute, a list of indexes that correspond to that value. 
        E.g. A list of indices for each gender in the dataset.

    Returns
    -------
    SP : float
        Statistical Parity score (0 to 1), calculated as the maximum rate of prediction y_hat minus minimum
        rate of prediction across all values v of the sensitive attribute X. Lower is better.  
        $\Delta_{SP} = \max_{v\in X}(P(\hat{y}=1 | X=v)) - \min_{v\in X}(P(\hat{y}=1 | X=v))$.
    """
    rates = []
    for v in sensitive_var_dict:
        rates.append(np.mean(pred[sensitive_var_dict[v]]))
    return max(rates) - min(rates)

def equality_of_odds(pred, true, sensitive_var_dict):
    """
    Equality of odds of model prediction across different sensitive attribute values for nodes that are outliers

    Parameters
    ----------
    pred : numpy.ndarray
        BINARY Outlier predictions (0 or 1) in shape of ``(N, )``where 1 represents outliers,
        0 represents normal instances.

    true : numpy.ndarray
        Labels in shape of ``(N, )``, where 1 represents outliers, 0 represents normal instances.

    sensitive_var_dict: dictionary of key -> numpy.ndarray
        For each value of the sensitive attribute, a list of indexes that correspond to that value. 
        E.g. A list of indices for each gender in the dataset.

    Returns
    -------
    EO : float
        Equality of odds score (0 to 1), calculated as the maximum rate of prediction y_hat minus minimum
        rate of prediction across all values v of the sensitive attribute X that are labelled as outliers.
        Lower is better.  
        $\Delta_{EO} = \max_{v\in X}(P(\hat{y}=1 | X=v, y=1)) - \min_{v\in X}(P(\hat{y}=1 | X=v, y=1))$.
    """
    sens_var = np.zeros_like(pred)
    number_of_values = len(sensitive_var_dict)
    for v in range(number_of_values):
        sens_var[sensitive_var_dict[v]] = v

    full = np.column_stack((pred, true, sens_var))
    only_true = full[full[:, 1] == 1]

    rates = []
    for v in range(number_of_values):
        temp = only_true[only_true[:, 2] == v]
        rates.append(np.mean(temp[:, 0]))

    if len(rates) < 2:
        warnings.warn("Positive labels only exist in <2 sensitive variable categories. EO value meaningless")

    return max(rates) - min(rates)
