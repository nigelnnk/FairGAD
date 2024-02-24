# -*- coding: utf-8 -*-
"""A set of utility functions to support outlier detection.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import requests
import warnings
import torch
import numpy as np
import numbers
import shutil

MAX_INT = np.iinfo(np.int32).max
MIN_INT = -1 * MAX_INT


def validate_device(gpu_id):
    """Validate the input device id (GPU id) is valid on the given
    machine. If no GPU is presented, return 'cpu'.

    Parameters
    ----------
    gpu_id : int
        GPU id to be used. The function will validate the usability
        of the GPU. If failed, return device as 'cpu'.

    Returns
    -------
    device_id : str
        Valid device id, e.g., 'cuda:0' or 'cpu'
    """
    # if it is cpu
    if gpu_id == -1:
        return 'cpu'

    # cast to int for checking
    gpu_id = int(gpu_id)

    # if gpu is available
    if torch.cuda.is_available():
        # check if gpu id is between 0 and the total number of GPUs
        check_parameter(gpu_id, 0, torch.cuda.device_count(), param_name='gpu id', include_left=True,
                        include_right=False)
        device_id = 'cuda:{}'.format(gpu_id)
    else:
        if gpu_id != 'cpu':
            warnings.warn('The cuda is not available. Set to cpu.')
        device_id = 'cpu'

    return device_id


def check_parameter(param, low=MIN_INT, high=MAX_INT, param_name='',
                    include_left=False, include_right=False):
    """Check if an input is within the defined range.
    Parameters
    ----------
    param : int, float
        The input parameter to check.
    low : int, float
        The lower bound of the range.
    high : int, float
        The higher bound of the range.
    param_name : str, optional (default='')
        The name of the parameter.
    include_left : bool, optional (default=False)
        Whether includes the lower bound (lower bound <=).
    include_right : bool, optional (default=False)
        Whether includes the higher bound (<= higher bound).
    Returns
    -------
    within_range : bool or raise errors
        Whether the parameter is within the range of (low, high)
    """

    # param, low and high should all be numerical
    if not isinstance(param, (numbers.Integral, np.integer, float)):
        raise TypeError('{param_name} is set to {param} Not numerical'.format(
            param=param, param_name=param_name))

    if not isinstance(low, (numbers.Integral, np.integer, float)):
        raise TypeError('low is set to {low}. Not numerical'.format(low=low))

    if not isinstance(high, (numbers.Integral, np.integer, float)):
        raise TypeError('high is set to {high}. Not numerical'.format(
            high=high))

    # at least one of the bounds should be specified
    if low is MIN_INT and high is MAX_INT:
        raise ValueError('Neither low nor high bounds is undefined')

    # if wrong bound values are used
    if low > high:
        raise ValueError(
            'Lower bound > Higher bound')

    # value check under different bound conditions
    if (include_left and include_right) and (param < low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (include_left and not include_right) and (
            param < low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and include_right) and (
            param <= low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and not include_right) and (
            param <= low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))
    else:
        return True


def load_data(name, cache_dir=None):
    """
    Data loading function. See `data repository
    <https://github.com/pygod-team/data>`_ for supported datasets.
    For injected/generated datasets, the labels meanings are as follows.

    - 0: inlier
    - 1: contextual outlier only
    - 2: structural outlier only
    - 3: both contextual outlier and structural outlier

    Parameters
    ----------
    name : str
        The name of the dataset.
    cache_dir : str, optional
        The directory for dataset caching.
        Default: ``None``.

    Returns
    -------
    data : torch_geometric.data.Data
        The outlier dataset.

    Examples
    --------
    >>> from pygod.utils import load_data
    >>> data = load_data(name='weibo') # in PyG format
    >>> y = data.y.bool()    # binary labels (inlier/outlier)
    >>> yc = data.y >> 0 & 1 # contextual outliers
    >>> ys = data.y >> 1 & 1 # structural outliers
    """

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.pygod/data')
    file_path = os.path.join(cache_dir, name+'.pt')
    zip_path = os.path.join(cache_dir, name+'.pt.zip')

    if os.path.exists(file_path):
        data = torch.load(file_path)
    else:
        url = "https://github.com/pygod-team/data/raw/main/" + name + ".pt.zip"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        shutil.unpack_archive(zip_path, cache_dir)
        data = torch.load(file_path)
    return data

def calculate_sp_loss(recon_error, sens_values, normalise=False):
    """
    This is the FairOD loss.
    """

    recon_err_mean = torch.mean(recon_error)
    recon_err_std = torch.sqrt(torch.var(recon_error, unbiased=False))
    recon_err_centered = (torch.sum(recon_error) - recon_err_mean) / recon_err_std

    sens_var_mean = torch.mean(sens_values)
    sens_var_std = torch.sqrt(torch.var(sens_values, unbiased=False))
    sens_var_centered = (torch.sum(sens_values) - sens_var_mean) / sens_var_std

    sp_loss = torch.abs(recon_err_centered * sens_var_centered)

    if normalise:
        sp_loss /= recon_error.size()[0]

    return sp_loss

def calculate_approx_ndcg_loss(recon_error, sens_values, base_scores, idcg_0, idcg_1):
    """
    This is the ADCG loss. 
    """
    idcg_val = [idcg_0, idcg_1]

    def idx_to_loss(idx):  
        sens_idx = (sens_values == idx).int().squeeze().nonzero(as_tuple=True)[0]
        curr_recon_error = torch.unsqueeze(torch.index_select(recon_error, 0, sens_idx), dim=1)
        num = sens_idx.shape[0]
        dist_between = curr_recon_error.repeat(1, num) - curr_recon_error.transpose(0,1).repeat(num, 1)
        denom = idcg_val[idx] * torch.log2(1 + torch.sum(torch.sigmoid(dist_between), axis=0))
        curr_base_error = torch.index_select(base_scores, 0, sens_idx)
        loss = 1 - torch.sum((torch.pow(2, curr_base_error) - 1) / denom)
        return loss
    
    return idx_to_loss(0) + idx_to_loss(1)

def hin_sp_loss(recon_error, sens_values, contamination):
    """
    This is the HIN loss.
    """
    expected_outlier_num = int(recon_error.shape[0] * contamination)
    sorted_err, indices = torch.sort(recon_error)
    sorted_sens = torch.zeros_like(sorted_err).scatter_(0, indices, sens_values)
    err_list = [sorted_err[:-expected_outlier_num], sorted_err[-expected_outlier_num:]]
    outliers_sens = [sorted_sens[:-expected_outlier_num], sorted_sens[-expected_outlier_num:]]

    loss = []
    for is_outlier in range(len(err_list)):
        temp = []
        for idx in [0, 1]:
            curr_idx = (outliers_sens[is_outlier] == idx).int().squeeze().nonzero(as_tuple=True)[0]
            curr_err = torch.index_select(err_list[is_outlier], 0, curr_idx)
            temp.append(curr_err.mean())
        loss.append(torch.square(temp[0]-temp[1])) 
    return loss[0] + loss[1]

def cor_sp_loss(loss, sens_values):
    """
    This is the Correlation loss. 
    """
    centered_loss = loss - loss.mean()
    centered_sens = sens_values - sens_values.mean()
    mag_loss = torch.linalg.norm(centered_loss)
    mag_sens = torch.linalg.norm(centered_sens)
    return torch.abs(torch.dot(centered_loss, centered_sens) / (mag_loss * mag_sens))
