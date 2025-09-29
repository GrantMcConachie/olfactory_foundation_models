"""
Adapted from https://github.com/AlexanderKroll/ProSmith
"""

import os
from os.path import join
import shutil
import random
import argparse
import time
import logging
import numpy as np
from time import gmtime, strftime
import pandas as pd
import pickle as pkl
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, rand
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    r2_score,
    mean_squared_error,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
from lifelines.utils import concordance_index


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model.MPP.utils.modules import (
    MM_TN,
    MM_TNConfig)

from model.MPP.utils.datautils import SMILESProteinDataset
from model.MPP.utils.train_utils import *


#### added this, fix the way argparse handles bool inputs: 
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#end of add

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oned",
        type=str2bool,
        help='specifies if input is a 1d vector or matrix of (seq_len x emb)',
        default=False
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="The input train dataset",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        required=False,
        help="The input val dataset",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=False,
        help="The input val dataset",
    )
    parser.add_argument(
        "--embed_path",
        type=str,
        required=True,
        help="Path that contains subfolders SMILES and Protein with embedding dictionaries",
    )
    parser.add_argument(
        "--save_pred_path",
        type=str,
        default="",
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--binary_task",
        default=False,
        type=str2bool, ##changed this to use str2bool func to properly handle bool values,
        help="Specifies wether the target variable is binary or continous.",
    )
    parser.add_argument(
        "--num_iter",
        default=2000,
        type=int,
        help="Total number of iterations to search for best set of hyperparameters.",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        required=False,
        help="Path of trained Transformer Network.",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=6,
        type=int,
        help="The num_hidden_layers size of MM_TN",
    )
    parser.add_argument(
        '--port',
        default=12557,
        type=int,
        help='Port for tcp connection for multiprocessing'
    )
    parser.add_argument(
        '--log_name',
        default="",
        type=str,
        help='Will be added to the file name of the log file'
    )
    parser.add_argument(
        '--dim_size',
        default=724,
        type=int,
        help='dimension of chemical embedding'
    )
    return parser.parse_args()

args = get_arguments()


n_gpus = len(list(range(torch.cuda.device_count())))


###########################################################################


def extract_repr(args, model, dataloader, device):
    print("device: %s" % device)
    # evaluate the model on validation set
    model.eval()
    logging.info(f"Extracting repr")

    if is_cuda(device):
        model = model.to(device)

    if "M2OR_full" in args.train_dir:
        criterion = M2ORWeightedCrossEntropyLoss(
            args.train_dir
        )
        weights = []

    else:
        weights = None
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            # move batch to device
            smiles_emb, smiles_attn, protein_emb, protein_attn, labels, indices, prots, smis = batch
            smiles_emb.to(device)
            smiles_attn.to(device)
            protein_emb.to(device)
            protein_attn.to(device)
            labels.to(device)
            indices.to(device)
            _, cls_repr = model(smiles_emb=smiles_emb, 
                                                    smiles_attn=smiles_attn, 
                                                    protein_emb=protein_emb,
                                                    protein_attn=protein_attn,
                                                    device=device,
                                                    gpu=0,
                                                    get_repr=True)

            protein_attn = int(sum(protein_attn.cpu().detach().numpy()[0]))
            smiles_attn = int(sum(smiles_attn.cpu().detach().numpy()[0]))

            smiles = smiles_emb[0][:smiles_attn].mean(0).cpu().detach().numpy()
            esm1b = protein_emb[0][:protein_attn].mean(0).cpu().detach().numpy()
            cls_rep = cls_repr[0].cpu().detach().numpy()

            if "M2OR_full" in args.train_dir:
                for i in range(len(prots)):
                    # assign weights to the data if using the m2or dataset
                    w_quality = criterion.data_quality_w[(smis[i], prots[i])]
                    w_class = criterion.pos_class_weight if labels[i] else criterion.neg_class_weight
                    w_pair = criterion.pair_imbalance_weight(smis[i], prots[i])

                    # get total weighting append to weights
                    w_tot = w_quality * w_class * w_pair
                    weights.append(w_tot)

            if step ==0:
                cls_repr_all = cls_rep.reshape(1,-1)
                esm1b_repr_all = esm1b.reshape(1,-1)
                smiles_repr_all = smiles.reshape(1,-1)
                labels_all = labels[0]
                logging.info(indices.cpu().detach().numpy())
                orginal_indices = list(indices.cpu().detach().numpy())
            else:
                cls_repr_all = np.concatenate((cls_repr_all, cls_rep.reshape(1,-1)), axis=0)
                smiles_repr_all = np.concatenate((smiles_repr_all, smiles.reshape(1,-1)), axis=0)
                esm1b_repr_all = np.concatenate((esm1b_repr_all, esm1b.reshape(1,-1)), axis=0)
                labels_all = torch.cat((labels_all, labels[0]), dim=0)
                orginal_indices = orginal_indices + list(indices.cpu().detach().numpy())

    # added this logging info step so you know if somethings getting stuck:
    logging.info("All batches processed. Representation extraction complete")
    #end add

    if "M2OR_full" in args.train_dir:
        weights = torch.tensor(weights, dtype=torch.float, device='cpu')

    return cls_repr_all, esm1b_repr_all, smiles_repr_all, labels_all.cpu().detach().numpy(), orginal_indices, weights



depth_array = [6,7,8,9,10,11,12,13,14]
space_gradient_boosting = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
    "max_depth": hp.choice("max_depth", depth_array),
    "reg_lambda": hp.uniform("reg_lambda", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 5),
    "max_delta_step": hp.uniform("max_delta_step", 0, 5),
    "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
    "num_rounds":  hp.uniform("num_rounds", 30, 1000),
    "weight" : hp.uniform("weight", 0.01,0.99)}


def trainer(gpu, args, device, par_dir, split, end_pth):
    ## logger
    # Create the log file path inside the directory
    log_dir = os.path.join(par_dir, split, end_pth)
    log_filename = (f'{args.log_name}_log.txt')
    log_path = os.path.join(log_dir, log_filename)

    # Clear existing handlers
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # File handler
    fhandler = logging.FileHandler(filename=log_path, mode='w')
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.info(f'split {split}')
    logger.info(args)

    if is_cuda(device):
        setup(gpu, args.world_size, str(args.port))
        torch.manual_seed(0)
        torch.cuda.set_device(gpu)
    
    # getting smiles dimension
    if args.oned:
        dim_size = list(pkl.load(open(args.embed_path, 'rb')).values())[0].shape[0]
    else:
        dim_size = list(pkl.load(open(args.embed_path, 'rb')).values())[0].shape[1]

    logger.info(f'dim size: {dim_size}')
    config = MM_TNConfig.from_dict({"s_hidden_size":dim_size, # NOTE: changing to accept mordred embeddings
        "p_hidden_size":1280,
        "hidden_size": 768,
        "max_seq_len":1276,
        "num_hidden_layers": args.num_hidden_layers,
        "binary_task": args.binary_task})


    logger.info(f"Loading dataset to {device}:{gpu}")
    train_dataset = SMILESProteinDataset(
        data_path=os.path.join(args.train_dir, split, 'train_df.csv'),
        embed_dir = args.embed_path,
        train=True,
        device=device, 
        gpu=gpu,
        random_state=0,
        binary_task=args.binary_task,
        extraction_mode=True,
        oned=args.oned) 

    val_dataset = SMILESProteinDataset(
        data_path=os.path.join(args.train_dir, split, 'val_df.csv'),
        embed_dir = args.embed_path,
        train=False, 
        device=device, 
        gpu=gpu,
        random_state=0,
        binary_task=args.binary_task,
        extraction_mode=True,
        oned=args.oned)
        
    test_dataset = SMILESProteinDataset(
        data_path=os.path.join(args.train_dir, split, 'test_df.csv'),
        embed_dir = args.embed_path,
        train=False, 
        device=device, 
        gpu=gpu,
        random_state=0,
        binary_task=args.binary_task,
        extraction_mode=True,
        oned=args.oned)

    trainsampler = DistributedSampler(train_dataset, shuffle = False, num_replicas = args.world_size, rank = gpu, drop_last = True)
    valsampler = DistributedSampler(val_dataset, shuffle = False, num_replicas = args.world_size, rank = gpu, drop_last = True)
    testsampler = DistributedSampler(test_dataset, shuffle = False, num_replicas = args.world_size, rank = gpu, drop_last = True)

    logger.info(f"Loading dataloader")
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1, sampler=trainsampler)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, sampler=valsampler)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, sampler=testsampler)


    logger.info(f"Loading model")
    model = MM_TN(config)
    
    if is_cuda(device):
        model = model.to(gpu)
        model = DDP(model, device_ids=[gpu])

    # geting pretrained model for this fold
    for file in os.listdir(log_dir):
        if file.endswith('.pkl'):
            pretrained_model = os.path.join(log_dir, file)

    if os.path.exists(pretrained_model) and pretrained_model != "":
        logger.info(f"Loading model")
        state_dict = torch.load(pretrained_model, map_location=device)
        new_model_state_dict = model.state_dict()
        try:
            for key in new_model_state_dict.keys():
                new_model_state_dict[key].copy_(state_dict[key])
                logger.info("Updatete key: %s" % key)

            model.load_state_dict(new_model_state_dict)
            logger.info("Successfully loaded pretrained model")
        except:
            new_state_dict = {}
            for key, value in state_dict.items():
                new_state_dict[key.replace("module.", "")] = value
            model.load_state_dict(new_state_dict)
            logger.info("Successfully loaded pretrained model (V2)")

    else:
        logger.info("Model path is invalid, cannot load pretrained MM_TN model")
    

    val_cls, val_esm1b, val_smiles, val_labels, _, weights_val = extract_repr(args, model, valloader, device)
    test_cls, test_esm1b, test_smiles, test_labels, test_indices, weights_test = extract_repr(args, model, testloader, device)
    train_cls, train_esm1b, train_smiles, train_labels, _, weights_train = extract_repr(args, model, trainloader, device)

    logger.info(str(len(test_labels)))
    
    logger.info(f"Extraction complete")
    
    def get_predictions(param, dM_train, dM_val, model_path=''):
        param, num_round, dM_train = set_param_values_V2(param = param, dtrain = dM_train)
        bst = xgb.train(param,  dM_train, num_round)

        # saving xgboost params
        if model_path == '':
            pass
        else:
            if os.path.isdir(model_path):
                bst.save_model(model_path)
            else:
                os.makedirs(os.path.dirname(model_path))
                bst.save_model(model_path)

        y_val_pred = bst.predict(dM_val)
        return(y_val_pred)
        
    def get_performance_metrics(pred, true):
        if args.binary_task:
            acc = np.mean(np.round(pred) == np.array(true))
            roc_auc = roc_auc_score(np.array(true), pred)
            mcc = matthews_corrcoef(np.array(true),np.round(pred))
            ave_p = average_precision_score(np.array(true), np.round(pred))
            prec = precision_score(np.array(true), np.round(pred))
            rec = recall_score(np.array(true), np.round(pred))
            f_score = f1_score(np.array(true), np.round(pred))
            logger.info("accuracy: %s,ROC AUC: %s, MCC: %s, AveP: %s, Prec: %s, rec: %s, f1: %s" % (acc, roc_auc, mcc, ave_p, prec, rec, f_score))
        else:
            mse = mean_squared_error(true, pred)
            CI = concordance_index(true, pred)
            rm2 = get_rm2(ys_orig = true, ys_line = pred)
            R2 = r2_score(true, pred)
            logger.info("MSE: %s,R2: %s, rm2: %s, CI: %s" % (mse, R2, rm2, CI))
    
    def set_param_values(param):
        num_round = int(param["num_rounds"])
        param["tree_method"] = "gpu_hist"
        param["sampling_method"] = "gradient_based"
        if not args.binary_task:
            param['objective'] = 'reg:squarederror'
            weights = None
        else:
            param['objective'] = 'binary:logistic'
            weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
            dtrain.set_weight(weights)

        del param["num_rounds"]
        del param["weight"]
        return(param, num_round)
    
    def set_param_values_V2(param, dtrain):
        num_round = int(param["num_rounds"])
        param["max_depth"] = int(depth_array[param["max_depth"]])
        param["tree_method"] = "gpu_hist"
        param["sampling_method"] = "gradient_based"
        if not args.binary_task:
            param['objective'] = 'reg:squarederror'
            weights = None
        else:
            param['objective'] = 'binary:logistic'
            weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
            dtrain.set_weight(weights)
        del param["num_rounds"]
        del param["weight"]
        return(param, num_round, dtrain)

    def set_param_values_all_cls(param):
        num_round = int(param["num_rounds"])
        
        param["tree_method"] = "gpu_hist"
        param["sampling_method"] = "gradient_based"
        if not args.binary_task:
            param['objective'] = 'reg:squarederror'
            weights = None
        else:
            param['objective'] = 'binary:logistic'
            weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain_all_cls.get_label()])
            dtrain_all_cls.set_weight(weights)
        del param["num_rounds"]
        del param["weight"]
        return(param, num_round)

    def set_param_values_cls(param):
        num_round = int(param["num_rounds"])
        
        param["tree_method"] = "gpu_hist"
        param["sampling_method"] = "gradient_based"
        if not args.binary_task:
            param['objective'] = 'reg:squarederror'
            weights = None
        else:
            param['objective'] = 'binary:logistic'
            weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain_cls.get_label()])
            dtrain_cls.set_weight(weights)
        del param["num_rounds"]
        del param["weight"]
        return(param, num_round)
        




        
    def get_performance(pred, true):
        if args.binary_task:
            MCC = matthews_corrcoef(true, np.round(pred))
            return(-MCC)
        else:
            MSE = mean_squared_error(true, pred)
            return(MSE)

        


    ############# ESM1b+ChemBERTa2
    
    train_X_all = np.concatenate([train_esm1b, train_smiles], axis = 1)
    test_X_all = np.concatenate([test_esm1b, test_smiles], axis = 1)
    val_X_all = np.concatenate([val_esm1b, val_smiles], axis = 1)

    dtrain = xgb.DMatrix(np.array(train_X_all), label = np.array(train_labels).astype(float), weight=weights_train)
    dtest = xgb.DMatrix(np.array(test_X_all), label = np.array(test_labels).astype(float), weight=weights_test)
    dvalid = xgb.DMatrix(np.array(val_X_all), label = np.array(val_labels).astype(float), weight=weights_val)
    dtrain_val = xgb.DMatrix(np.concatenate([np.array(train_X_all), np.array(val_X_all)], axis = 0),
                                    label = np.concatenate([np.array(train_labels).astype(float),np.array(val_labels).astype(float)], axis = 0), weight=np.concatenate([weights_train, weights_val]))
   
    def train_xgboost_model_all(param):
        param, num_round = set_param_values(param)
        #Training:
        bst = xgb.train(param,  dtrain, num_round)
        return(get_performance(pred = bst.predict(dvalid), true =val_labels))

    trials = Trials()
    best = fmin(fn = train_xgboost_model_all, space = space_gradient_boosting,
                algo = rand.suggest, max_evals = args.num_iter, trials = trials)


    #predictions for validation and test set on test set:
    logger.info("ESM1b+ChemBERTa2")
    logger.info("Validation set:")
    y_val_pred_all = get_predictions(param = trials.argmin, dM_train = dtrain, dM_val = dvalid)
    get_performance_metrics(pred = y_val_pred_all, true = val_labels)
    logger.info("Test set:")
    y_test_pred_all = get_predictions(param = trials.argmin, dM_train = dtrain_val, dM_val = dtest, model_path=os.path.join(log_dir, 'xgboost', 'embs_only.model'))
    get_performance_metrics(pred = y_test_pred_all, true = test_labels)
    
    
    ############# ESM1b+ChemBERTa +cls
    train_X_all_cls = np.concatenate([np.concatenate([train_esm1b, train_smiles], axis = 1), train_cls], axis=1)
    test_X_all_cls = np.concatenate([np.concatenate([test_esm1b, test_smiles], axis = 1), test_cls], axis=1)
    val_X_all_cls = np.concatenate([np.concatenate([val_esm1b, val_smiles], axis = 1), val_cls], axis=1)

    dtrain_all_cls = xgb.DMatrix(np.array(train_X_all_cls), label = np.array(train_labels).astype(float), weight=weights_train)
    dtest_all_cls = xgb.DMatrix(np.array(test_X_all_cls), label = np.array(test_labels).astype(float), weight=weights_test)
    dvalid_all_cls = xgb.DMatrix(np.array(val_X_all_cls), label = np.array(val_labels).astype(float), weight=weights_val)
    dtrain_val_all_cls = xgb.DMatrix(np.concatenate([np.array(train_X_all_cls), np.array(val_X_all_cls)], axis = 0),
                                label = np.concatenate([np.array(train_labels).astype(float),np.array(val_labels).astype(float)], axis = 0), weight=np.concatenate([weights_train, weights_val]))
    
    
    def train_xgboost_model_all_cls(param):
        param, num_round = set_param_values_all_cls(param)
        #Training:
        bst = xgb.train(param,  dtrain_all_cls, num_round)
        return(get_performance(pred = bst.predict(dvalid_all_cls), true =val_labels))


    trials = Trials()
    best = fmin(fn = train_xgboost_model_all, space = space_gradient_boosting,
                algo = rand.suggest, max_evals = args.num_iter, trials = trials)


    #predictions for validation and test set on test set
    logger.info("ESM1b+ChemBERTa2+cls-token")
    logger.info("Validation set:")
    y_val_pred_all_cls = get_predictions(param = trials.argmin, dM_train = dtrain_all_cls, dM_val = dvalid_all_cls)
    get_performance_metrics(pred = y_val_pred_all_cls, true = val_labels)
    logger.info("Test set:")
    y_test_pred_all_cls = get_predictions(param  = trials.argmin, dM_train = dtrain_val_all_cls, dM_val = dtest_all_cls, model_path=os.path.join(log_dir, 'xgboost', 'embs_and_cls.model'))
    get_performance_metrics(pred = y_test_pred_all_cls, true = test_labels)



    ############# cls token
    dtrain_cls = xgb.DMatrix(np.array(train_cls), label = np.array(train_labels).astype(float), weight=weights_train)
    dvalid_cls = xgb.DMatrix(np.array(val_cls), label = np.array(val_labels).astype(float), weight=weights_val)
    dtest_cls = xgb.DMatrix(np.array(test_cls), label = np.array(test_labels).astype(float), weight=weights_test)
    dtrain_val_cls = xgb.DMatrix(np.concatenate([np.array(train_cls), np.array(val_cls)], axis = 0),
                                label = np.concatenate([np.array(train_labels).astype(float),np.array(val_labels).astype(float)], axis = 0), weight=np.concatenate([weights_train, weights_val]))

    
    def train_xgboost_model_cls(param):
        param, num_round = set_param_values_cls(param)
        #Training:
        bst = xgb.train(param,  dtrain_cls, num_round)
        return(get_performance(pred = bst.predict(dvalid_cls), true =val_labels))


    trials = Trials()
    best = fmin(fn = train_xgboost_model_cls, space = space_gradient_boosting,
                algo = rand.suggest, max_evals = args.num_iter, trials = trials)
                
                
    #predictions for validation and test set on test set:
    logger.info("cls-token")
    logger.info("Validation set:")
    y_val_pred_cls = get_predictions(param = trials.argmin, dM_train = dtrain_cls, dM_val = dvalid_cls)
    get_performance_metrics(pred = y_val_pred_cls, true = val_labels)
    logger.info("Test set:")
    y_test_pred_cls = get_predictions(param = trials.argmin, dM_train = dtrain_val_cls, dM_val = dtest_cls, model_path=os.path.join(log_dir, 'xgboost', 'cls.model'))
    get_performance_metrics(pred = y_test_pred_cls, true = test_labels)



    #############
    best_mcc, best_mse = 0, 1000
    best_i, best_j, best_k = 0,0,0
    for i in [k/100 for k in range(0,100)]:
        for j in [k/100 for k in range(0,100)]:
            if i+j <=1:
                k = (1-i-j)
                y_val_pred = i*y_val_pred_all_cls + j*y_val_pred_all  + k*y_val_pred_cls
                if args.binary_task:
                    mcc = matthews_corrcoef(val_labels, np.round(y_val_pred))
                    if mcc > best_mcc:
                        best_mcc = mcc
                        best_i, best_j, best_k = i, j, k
                else:
                    mse = mean_squared_error(val_labels, y_val_pred)
                    if mse < best_mse:
                        best_mse = mse
                        best_i, best_j, best_k = i, j, k
    
    # saving best proportion
    pkl.dump((best_i, best_j, best_k), open(os.path.join(log_dir, 'xgboost', 'val_proportion.pkl'), 'wb'))

    y_test_pred = best_i*y_test_pred_all_cls + best_j*y_test_pred_all + best_k*y_test_pred_cls
    logger.info("Three models combined:")
    logger.info("ESM1b+ChemBERTa2+cls: %s, ESM1b+ChemBERTa2: %s, cls-token: %s" %(best_i, best_j, best_k))
    get_performance_metrics(pred = y_test_pred, true = test_labels)

    

    ###Save model predictions:
    save_pred_path = os.path.join(os.path.dirname(par_dir), 'predictions', split, end_pth)
    try:
        os.makedirs(save_pred_path)
    except:
        logger.info('Predictions did not save!') 
    np.save(join(save_pred_path, "y_test_pred.npy"), y_test_pred)
    np.save(join(save_pred_path, "test_indices.npy"), np.array(test_indices))

    if args.world_size != -1:
        cleanup()


def check_if_transformer_trained(args):
    """
    Automatically checks if a transformer was trained
    """
    # make sure only checking certain splits
    if args.train_dir.split('/')[-1] == 'rand_splits':
        split_type = 'rand_split'
    elif args.train_dir.split('/')[-1] == 'cdhit_splits':
        split_type = 'cdhit_split'
    elif args.train_dir.split('/')[-1] == 'scaf_splits':
        split_type = 'scaf_split'
    else:
        raise Exception(f'Choose appropriate split')

    par_dir = os.path.dirname(args.train_dir.replace('data', 'results'))
    par_dir = os.path.join(par_dir, 'MPP', 'saved_models')
    end_pth, _ = os.path.splitext(os.path.basename(args.embed_path))

    for i in os.listdir(par_dir):
        _dir = os.path.join(par_dir, i, end_pth)
        if split_type in _dir:
            if os.path.isdir(_dir):
                pass
            else:
                raise Exception(f'Train transformer on embedding first')
        
    return par_dir, end_pth


if __name__ == '__main__':
    # Set up the device
    
    # Check if multiple GPUs are available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_ids = list(range(torch.cuda.device_count()))
        gpus = len(device_ids)
        args.world_size = gpus
        
    else:
        device = torch.device('cpu')
        args.world_size = -1

    par_dir, end_pth = check_if_transformer_trained(args)

    # Loop through folds
    for split in os.listdir(args.train_dir):

        try:
            if torch.cuda.is_available():
                mp.spawn(trainer, nprocs=args.world_size, args=(args, device, par_dir, split, end_pth))
            else:
                trainer(0, args, device, par_dir, split, end_pth)
        except Exception as e:
            print(e)
