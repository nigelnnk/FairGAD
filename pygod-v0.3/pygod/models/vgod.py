# Unsupervised Graph Outlier Detection: Problem Revisit, New Insight, and Superior Method
# Base code from: https://github.com/goldenNormal/vgod-github

import time
import warnings

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch_geometric.nn import GIN, MessagePassing
from torch_geometric.typing import OptTensor, OptPairTensor
from torch_geometric.utils import to_dense_adj, negative_sampling
from scipy.stats import rankdata

from . import BaseDetector
from ..utils import validate_device, calculate_sp_loss, calculate_approx_ndcg_loss, hin_sp_loss, cor_sp_loss
from ..metrics import eval_roc_auc


class VGOD(BaseDetector):
    def __init__(self, gpu, verbose, contamination,
                 alpha=1, emb_dim=128, structural_epoch=10, epoch=100, lr=0.005, weight_decay=0.0001,
                 ):
        super(VGOD, self).__init__(contamination=contamination)

        self.emb_dim = emb_dim
        self.alpha = alpha

        self.weight_decay = weight_decay
        self.lr = lr
        self.str_epoch = structural_epoch
        self.epoch = epoch
        self.device = validate_device(gpu)
        self.verbose = verbose

        self.str_model = None
        self.att_model = None

    def fit(self, data):
        input_dim, att, edge_index = self.process_graph(data)
        str_model = NeiVar(input_dim, self.emb_dim).to(self.device)
        att_model = Recon(input_dim, self.emb_dim, GIN).to(self.device)
        str_opt = Adam(str_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        att_opt = Adam(att_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        var_loss = 0
        for epoch in range(self.epoch):
            if self.str_epoch > epoch:
                str_model.train()
                neg_edge = negative_sampling(edge_index, num_neg_samples=edge_index.shape[1])
                pos_loss = str_model(att, edge_index)
                neg_loss = str_model(att, neg_edge)
                var_loss = torch.mean(pos_loss) - torch.mean(neg_loss)

            att_model.train()
            recon_loss = att_model(att, edge_index)
            mean_recon_loss = torch.mean(recon_loss)

            epoch_loss = mean_recon_loss + self.alpha * var_loss
            decision_scores = self.add_two_score_std(pos_loss, recon_loss)

            if isinstance(self.verbose, int) and self.verbose > 1:
                print(f"\tPos:{pos_loss}\tNeg:{neg_loss}\n\tVar  :{var_loss}\n" + 
                      f"\tRecon:{recon_loss}\nEpoch:{epoch_loss}")

            if self.str_epoch <= epoch:
                att_opt.zero_grad()
                mean_recon_loss.backward()
                att_opt.step()
            else:
                att_opt.zero_grad()
                str_opt.zero_grad()
                epoch_loss.backward()
                att_opt.step()
                str_opt.step()

        self.decision_scores_ = decision_scores.detach().cpu().numpy()
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

        input_dim, att, edge_index = self.process_graph(data)
        str_model = NeiVar(input_dim, self.emb_dim).to(self.device)
        att_model = Recon(input_dim, self.emb_dim, GIN).to(self.device)
        str_opt = Adam(str_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        att_opt = Adam(att_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        var_loss = 0
        curr_sensitive = sens_var_col.to(self.device)
        for epoch in range(self.epoch):
            if self.str_epoch > epoch:
                str_model.train()
                neg_edge = negative_sampling(edge_index, num_neg_samples=edge_index.shape[1])
                pos_loss = str_model(att, edge_index)
                neg_loss = str_model(att, neg_edge)
                var_loss = torch.mean(pos_loss) - torch.mean(neg_loss)

            att_model.train()
            recon_loss = att_model(att, edge_index)
            mean_recon_loss = torch.mean(recon_loss)

            epoch_loss = mean_recon_loss + self.alpha * var_loss
            score = self.add_two_score_std(pos_loss, recon_loss)

            if sens_var_col is not None and self.str_epoch > epoch:
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
                    print(f"Loss: {epoch_loss}\nSP Loss: {sp_loss}\nADCG Loss: {approx_ndcg_loss}")
                epoch_loss = epoch_loss + fair_factor * sp_loss + adcg_factor * approx_ndcg_loss

            if isinstance(self.verbose, int) and self.verbose > 1:
                print(f"\tPos:{pos_loss}\tNeg:{neg_loss}\n\tVar  :{var_loss}\n" + 
                      f"\tRecon:{recon_loss}\nEpoch:{epoch_loss}")

            if self.str_epoch <= epoch:
                att_opt.zero_grad()
                mean_recon_loss.backward()
                att_opt.step()
            else:
                att_opt.zero_grad()
                str_opt.zero_grad()
                epoch_loss.backward()
                att_opt.step()
                str_opt.step()

        self.decision_scores_ = score.detach().cpu().numpy()
        self._process_decision_scores()
        return self

    def decision_function(self, G):
        outlier_scores = self.decision_scores_
        return outlier_scores
    
    def process_graph(self, G):
        att = G.x.to(self.device)
        edge_index = G.edge_index.to(self.device)
        input_dim = att.shape[1]
        return input_dim, att, edge_index


    def std_scale(self, x):
        x = (x - torch.mean(x))/torch.std(x)
        return x

    def add_two_score_std(self, score1,score2):
        score1 = self.std_scale(score1)
        score2 = self.std_scale(score2)
        return score1 + score2 * self.alpha



class MeanConv(MessagePassing):

    def __init__(self, aggr: str = 'mean', **kwargs):
        super().__init__(aggr=aggr, **kwargs)

    def forward(self, x, edge_index,
                edge_weight: OptTensor = None, size: torch.Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        return out

    def message(self, x_j: Tensor,x_i:Tensor) -> Tensor:
        return x_j

class CovConv(MessagePassing):

    def __init__(self, aggr: str = 'mean', **kwargs):
        super().__init__(aggr=aggr, **kwargs)

    def forward(self, x, edge_index,ner_mean,
                edge_weight: OptTensor = None, size: torch.Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, ner_mean = ner_mean[edge_index[1]],edge_weight=edge_weight,
                             size=size)
        out = torch.sum(out,dim=-1)
        return out

    def message(self, x_j: Tensor,ner_mean) -> Tensor:
        return (x_j - ner_mean)**2

class NeiVar(nn.Module):
    def __init__(self,input_dim,emb_dim):
        super(NeiVar, self).__init__()
        self.lin = nn.Linear(input_dim,emb_dim)
        self.mean = MeanConv()
        self.cov = CovConv()

    def forward(self,x,edge_index):
        h = self.lin(x)
        h = h/(torch.norm(h,dim=-1).reshape(-1,1))
        mean = self.mean(h,edge_index)
        var = self.cov(h,edge_index,mean)
        return var

class Recon(nn.Module):
    def __init__(self,input_dim,emb_dim,GNN):
        super(Recon, self).__init__()
        self.lin = nn.Linear(input_dim,emb_dim)
        self.gnn = GNN(emb_dim,emb_dim,2)
        self.lin2 = nn.Linear(emb_dim,input_dim)

    def forward(self,x,edge_index):
        h = self.lin(x)
        h = h/(torch.norm(h,dim=-1).reshape(-1,1))
        recon_x = self.gnn(h,edge_index)
        recon_x = self.lin2(recon_x)
        return torch.sum(torch.square(x - recon_x),dim=-1)
