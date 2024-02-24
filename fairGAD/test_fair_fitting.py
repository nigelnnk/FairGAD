import argparse
import numpy as np
import torch
import tqdm
import gc

from torch_geometric.seed import seed_everything
from pygod.metrics import eval_roc_auc, statistical_parity, equality_of_odds
from pygod.models import CONAD, CoLA, DOMINANT, AdONE, DONE, ONE_NEW, VGOD

from sklearn.metrics import precision_recall_curve, auc
from scipy.special import erf

MODELS = ['COLA', 'CONAD', 'DOMINANT', "ADONE", "DONE", "ONE", "VGOD"]
REGULARISERS = ["hin", "fairod", "correlation", "none"]


def sensitive_tensor_to_idx_dict(tensor):
    values, inverse_indices = torch.unique(tensor, return_inverse=True)
    values = values.tolist()
    idx_dict = dict()

    for i, v in enumerate(values):
        idx_dict[v] = (inverse_indices == i).nonzero()
    
    return idx_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
        default='DOMINANT',
        help='GOD model from {COLA, CONAD, DOMINANT, ONE, VGOD}')
    parser.add_argument('--data',
        default='fairgad_reddit',
        help='Dataset evaluation from {our_reddit, our_twitter}')
    parser.add_argument('--regulariser',
        default="hin", type=str,
        help='Type of fairness regulariser to use')
    parser.add_argument('--fair_factor',
        default=0, type=float,
        help='Factor to weight fair fitting procedure (default is 0)')
    parser.add_argument('--adcg_factor',
        default=0, type=float,
        help='Factor to weight ndcg component (default is 0)')
    parser.add_argument('--num_trials',
        default=20, type=int,
        help='Number of times to repeat experiment')
    parser.add_argument('--seed',
        default=1, type=int,
        help='Random seed')
    parser.add_argument('--gpu',
        default=3, type=int,
        help='GPU to use')
    parser.add_argument('--batch_size',
        default=16384, type=int,
        help='Size of batch to use')
    parser.add_argument('--verbose',
        default=0, type=int,
        help='Verbosity')
    args = parser.parse_args()

    seed_everything(args.seed)
    DEVICE = args.gpu
    BATCH_SIZE = args.batch_size

    assert args.model in MODELS, "Invalid model chosen"
    assert args.regulariser in REGULARISERS, "Invalid regulariser chosen"

    torch.set_num_threads(12)

    if args.data == "our_reddit":
        data = torch.load(f"../../data/reddit.pt")
    elif args.data == "our_twitter":
        data = torch.load(f"../../data/twitter.pt")
    elif "EDITS" in args.data:
        data = torch.load(f"../../data/{args.data}_epoch_500.pt")
    else:
        data = torch.load(f"../../data/{args.data}.pt")
    
    data.y = data.y.bool()
    data.x = data.x.float()
    contamination = data.contamination
    data.sensitive = data.sensitive.float()
    sensitive_dict = sensitive_tensor_to_idx_dict(data.sensitive)

    aucs = []
    sps = []
    eos = []
    auprcs = []
    nan = 0
    for _ in tqdm.tqdm(range(args.num_trials)):
        if args.model == 'DOMINANT':
            model = DOMINANT(batch_size=BATCH_SIZE, gpu=DEVICE, contamination=contamination, verbose=args.verbose)
        elif args.model == 'CONAD':
            model = CONAD(batch_size=BATCH_SIZE, gpu=DEVICE, contamination=contamination, verbose=args.verbose)
        elif args.model == 'COLA':
            model = CoLA(batch_size=BATCH_SIZE, gpu=DEVICE, contamination=contamination, verbose=args.verbose)
        elif args.model == 'ADONE':
            model = AdONE(batch_size=BATCH_SIZE, gpu=DEVICE, contamination=contamination, verbose=args.verbose)
        elif args.model == 'DONE':
            model = DONE(batch_size=BATCH_SIZE, gpu=DEVICE, contamination=contamination, verbose=args.verbose)
        elif args.model == 'ONE':
            model = ONE_NEW(gpu=DEVICE, contamination=contamination, verbose=args.verbose)
        elif args.model == 'VGOD':
            model = VGOD(gpu=DEVICE, contamination=contamination, verbose=args.verbose)

        gc.collect()
        torch.cuda.empty_cache()

        if args.fair_factor > 0 or args.adcg_factor > 0:
            model.fit_with_fairness(data,
                                    fair_factor=args.fair_factor,
                                    sens_var_col=data.sensitive.float(),
                                    adcg_factor=args.adcg_factor,
                                    regulariser=args.regulariser)
        else:
            model.fit(data)

        gc.collect()
        torch.cuda.empty_cache()
        outlier_prob = model.predict_proba(data, method="unify")[:, 1]
        prediction = model.predict(data)

        gc.collect()
        torch.cuda.empty_cache()
        del model

        if np.any(np.isnan(outlier_prob)):
            nan += 1
            torch.cuda.empty_cache()
            continue

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
    
    # print(f"Regulariser|Fair Factor|ADCG Factor|Dataset|Model|AUCROC|SP|EO|AUPRC")
    print(f"{args.regulariser}|{args.fair_factor}|{args.adcg_factor}|{args.data}|{args.model}|" +
          f"{auc_score:.4f}±{auc_range:.4f}|" + 
          f"{sp_score:.4f}±{sp_range:.4f}|" +
          f"{eo_score:.4f}±{eo_range:.4f}|" + 
          f"{auprc_score:.4f}±{auprc_range:.4f}|" +
          f"{nan}")  # Count of NaN for each trial
