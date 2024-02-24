import json
import numpy as np
import torch
import tqdm
import gc

from torch_geometric.seed import seed_everything
from pygod.metrics import eval_roc_auc, statistical_parity, equality_of_odds

from sklearn.metrics import precision_recall_curve, auc

MODELS = ['COLA', 'CONAD', 'DOMINANT']
REGULARISERS = ["hin", "fairod", "correlation"]


def sensitive_tensor_to_idx_dict(tensor):
    values, inverse_indices = torch.unique(tensor, return_inverse=True)
    values = values.tolist()
    idx_dict = dict()

    for i, v in enumerate(values):
        idx_dict[v] = (inverse_indices == i).nonzero()
    
    return idx_dict


def benchmark(model, data, config_path):
    config = json.load(config_path)
    seed_everything(config["seed"])

    torch.set_num_threads(config["threads"])

    data.y = data.y.bool()
    data.x = data.x.float()
    data.sensitive = data.sensitive.float()
    sensitive_dict = sensitive_tensor_to_idx_dict(data.sensitive)

    aucs = []
    sps = []
    eos = []
    auprcs = []
    nan = 0
    for _ in tqdm.tqdm(range(config["num_trials"])):
        gc.collect()
        torch.cuda.empty_cache()
        model.fit_with_fairness(data, sens_var_col=data.sensitive, **config)  # reinitialises model each time

        torch.cuda.empty_cache()
        outlier_prob = model.predict_proba(data, method="unify")[:, 1]
        if np.any(np.isnan(outlier_prob)):
            nan += 1
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()
        prediction = model.predict(data)
        torch.cuda.empty_cache()
        aucs.append(eval_roc_auc(data.y.numpy(), outlier_prob))
        sps.append(statistical_parity(prediction, sensitive_dict))
        eos.append(equality_of_odds(prediction, data.y.numpy(), sensitive_dict))
        precision, recall, _ = precision_recall_curve(data.y.numpy(), outlier_prob)
        auprcs.append(auc(recall, precision))
        torch.cuda.empty_cache()

    auc_score = np.mean(aucs)
    auc_range = np.std(aucs)
    sp_score = np.mean(sps)
    sp_range = np.std(sps)
    eo_score = np.mean(eos)
    eo_range = np.std(eos)
    auprc_score = np.mean(auprcs)
    auprc_range = np.std(auprcs)

    print(f"Results:\n" +
          f"{auc_score:.4f}±{auc_range:.4f}|" + 
          f"{sp_score:.4f}±{sp_range:.4f}|" +
          f"{eo_score:.4f}±{eo_range:.4f}|" + 
          f"{auprc_score:.4f}±{auprc_range:.4f}|" +
          f"{nan}")  # Count of NaN for each trial

