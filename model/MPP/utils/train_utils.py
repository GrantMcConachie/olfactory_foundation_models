import os
import torch
import torch.distributed as dist
import logging
import numpy as np
import pandas as pd
import torch.nn as nn


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def is_cuda(device):
    return device == torch.device('cuda')


def accuracy(y_pred, y_true):
    # Calculate accuracy
    correct = (y_pred == y_true).sum().item()
    total = y_true.shape[0]
    acc = correct / total    
    return acc


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0
    
    
def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    ys_orig = np.array(ys_orig).reshape(-1)
    ys_line = np.array(ys_line).reshape(-1)
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

class M2ORWeightedCrossEntropyLoss(nn.Module):
    """
    Weighted loss function used in Hladiš et. al. to account for data quality
    for the M2OR dataset
    """
    def __init__(self, data_path):
        super(M2ORWeightedCrossEntropyLoss, self).__init__()

        # loading full data
        full_dat = os.path.join(
            os.path.dirname(data_path),
            'raw',
            'full_data.csv'
        )
        full_dat_df = pd.read_csv(full_dat)

        ## Data quality weights
        primary_pos_weight = 0.40
        primary_neg_weight = 0.69
        secondary_pos_weight = 0.72
        secondary_neg_weight = 0.77
        ec50_weight = 1.0

        # create list of conditions
        conditions = [
            full_dat_df['_DataQuality'] == 'ec50',
            (full_dat_df['_DataQuality'] == 'primaryScreening') & (full_dat_df['Responsive']),
            (full_dat_df['_DataQuality'] == 'primaryScreening') & (~full_dat_df['Responsive']),
            (full_dat_df['_DataQuality'] == 'secondaryScreening') & (full_dat_df['Responsive']),
            (full_dat_df['_DataQuality'] == 'secondaryScreening') & (~full_dat_df['Responsive']),
        ]

        # list of correcponding weights
        choices = [
            ec50_weight,
            primary_pos_weight,
            primary_neg_weight,
            secondary_pos_weight,
            secondary_neg_weight
        ]

        # create column of weights
        full_dat_df['data_quality_w'] = np.select(conditions, choices, default=np.nan)

        # Check for Incorrect data quality label
        if full_dat_df['data_quality_w'].isnull().any():
            raise Exception("Incorrect data quality label")

        # build weight dict
        self.data_quality_w = dict(
            zip(
                zip(full_dat_df['SMILES'], full_dat_df['Protein sequence']),
                full_dat_df['data_quality_w']
            )
        )

        ## calculating class imbalance weights
        num_responsive = (full_dat_df['Responsive'] == 1).sum()
        num_non_responsive = (full_dat_df['Responsive'] == 0).sum()
        self.pos_class_weight = num_non_responsive / num_responsive
        self.neg_class_weight = 1.0

        ## pair imbalance
        self.num_mol_per_prot = full_dat_df['Protein sequence'].value_counts().to_dict()
        self.num_prot_per_mol = full_dat_df['SMILES'].value_counts().to_dict()
        self.K = 100.0
    
    def pair_imbalance_weight(self, mol, prot):
        """
        Eq 5 in supplement C of Hladiš et. al.
        """
        w_pair = np.log(1 + self.K / 2 * (1/self.num_mol_per_prot[prot] + 1/self.num_prot_per_mol[mol]))
        return w_pair

    def forward(self, pred, target, smiles, proteins):
        batch_size = pred.shape[0]

        # Calculate a weight for each sample in the batch based on the SMILES and protein
        weights = []
        for i in range(batch_size):
            smi = smiles[i]
            prot = proteins[i]

            # get individual weights
            w_quality = self.data_quality_w[(smiles[i], proteins[i])]
            w_class = self.pos_class_weight if target[i] else self.neg_class_weight
            w_pair = self.pair_imbalance_weight(smi, prot)

            # get total weighting append to weights
            w_tot = w_quality * w_class * w_pair
            weights.append(w_tot)

        # Convert to tensor, same device/type as pred
        weights = torch.tensor(weights, dtype=pred.dtype, device=pred.device).unsqueeze(1)

        # Compute BCE loss per sample (no reduction)
        bce = nn.BCEWithLogitsLoss(weight=weights, reduction='mean')
        target = target.to(pred.device)
        loss = bce(pred, target)

        return loss