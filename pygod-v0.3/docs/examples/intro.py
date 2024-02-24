"""
Model Example
================

In this introductory tutorial, you will learn the basic workflow of
PyGOD with a model example of DOMINANT. This tutorial assumes that
you have basic familiarity with PyTorch and PyTorch Geometric (PyG).

(Time estimate: 5 minutes)
"""
#######################################################################
# Data Loading
# ------------
# PyGOD use `torch_geometric.data.Data` to handle the data. Here, we
# use Cora, a PyG built-in dataset, as an example. To load your own
# dataset into PyGOD, you can refer to [creating your own datasets
# tutorial](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html)
# in PyG.


import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

data = Planetoid('./data/Cora', 'Cora', transform=T.NormalizeFeatures())[0]

#######################################################################
# Because there is no ground truth label of outliers in Cora, we follow
# the method used by DOMINANT to inject 100 contextual outliers and 100
# structure outliers into the graph. **Note**: If your dataset already
# contains the outliers you want to detect, you don't need to inject
# more outliers.


import torch
from pygod.generator import gen_contextual_outliers, gen_structural_outliers

data, ya = gen_contextual_outliers(data, n=100, k=50)
data, ys = gen_structural_outliers(data, m=10, n=10)
data.y = torch.logical_or(ys, ya).int()


#######################################################################
# **New feature for PyGOD 0.3.0: we now provide built-in datasets!**
# See [data repository](https://github.com/pygod-team/data) for more
# details.

from pygod.utils import load_data

data = load_data('inj_cora')
data.y = data.y.bool()

#######################################################################
# Initialization
# --------------
# You can use any model by simply initializing without passing any
# arguments. Default hyperparameters are ready for you. Of course, you
# can also customize the parameters by passing arguments. Here, we use
# `pygod.models.DOMINANT` as an example.


from pygod.models import DOMINANT

model = DOMINANT()

#######################################################################
# Training
# --------
# To train the model with the loaded data, simply feed the
# `torch_geometric.data.Data` object into the model via method `fit`.


model.fit(data)

#######################################################################
# Inference
# ---------
# Then, your model is ready to use.
# We provide several inference methods.
#
# To predict the labels only:


labels = model.predict(data)
print('Labels:')
print(labels)

#######################################################################
# To predict raw outlier scores:


outlier_scores = model.decision_function(data)
print('Raw scores:')
print(outlier_scores)

#######################################################################
# To predict the probability of the outlierness:


prob = model.predict_proba(data)
print('Probability:')
print(prob)

#######################################################################
# To predict the labels with confidence:


labels, confidence = model.predict(data, return_confidence=True)
print('Labels:')
print(labels)
print('Confidence:')
print(confidence)

#######################################################################
# To evaluate the performance outlier detector:


from pygod.metrics import eval_roc_auc

auc_score = eval_roc_auc(data.y.numpy(), outlier_scores)
print('AUC Score:', auc_score)
