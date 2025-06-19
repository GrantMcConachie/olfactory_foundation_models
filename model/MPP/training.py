import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils.modules import (
    MM_TN,
    MM_TNConfig,
)

from utils.datautils import SMILESProteinDataset
from utils.train_utils import *

import os
import pandas as pd
import shutil
import random
import argparse
import time
import logging
import numpy as np
import pickle as pkl
from time import gmtime, strftime
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, precision_score, recall_score
from lifelines.utils import concordance_index


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
        "--embed_path",
        type=str,
        required=True,
        help="Path that contains subfolders SMILES and Protein with embedding dictionaries",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--batch_size",
        default=12,
        type=int,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--binary_task",
        default=False,
        type=str2bool, ##changed this to use str2bool func to properly handle bool values
        help="Specifies wether the target variable is binary or continous.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=50,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup",
    )
    parser.add_argument(
        "--pretrained_model",
        default='',
        type=str,
        help="Path of pretrained model. If empty model will be trained from scratch.",
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
        required=False,
        type=int,
        help='dimension of chemical embedding'
    )
    return parser.parse_args()

args = get_arguments()


n_gpus = len(list(range(torch.cuda.device_count())))
eff_bs =  n_gpus*args.batch_size
args.port = args.port

setting = args.log_name + '_' + str(n_gpus) +'gpus_bs' + str(eff_bs) +'_'+str(args.learning_rate) +'_layers' + str(args.num_hidden_layers) +'.txt'

###########################################################################

def train(args, model, trainloader, optimizer, criterion, device, gpu, epoch):
    model.train()
    train_loss = 0.
    
    logging.info(f"Training for epoch {epoch+1}")
    
    for step, batch in enumerate(trainloader):
        # logging.info(f"Batch: {step}, Time ={np.round(time.time()-start_time)}")
        if is_cuda(device):
                batch = [r.cuda(gpu) for r in batch]
        smiles_emb, smiles_attn, protein_emb, protein_attn, labels, _ = batch
        
        # zero the gradients
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(smiles_emb=smiles_emb, 
            smiles_attn=smiles_attn, 
            protein_emb=protein_emb,
            protein_attn=protein_attn,
            device=device, gpu=gpu)
        
        loss = criterion(outputs, labels.float())
        
        # backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # logging.info training loss after every n steps
        if step % 10 == 0:
            logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, args.num_train_epochs, step+1, len(trainloader), loss.item()))
        train_loss += loss.item()
        
    return train_loss / len(trainloader)

# !!! changed below:
def evaluate(args, model, valloader, criterion, device, gpu):
    # evaluate the model on validation set
    model.eval()
    y_true, y_pred = [], []
    val_loss = 0.
    
    logging.info(f"Evaluating")
    
    with torch.no_grad():
        for step, batch in enumerate(valloader):
            # move batch to device
            if is_cuda(device):
                batch = [r.cuda(gpu) for r in batch]
            smiles_emb, smiles_attn, protein_emb, protein_attn, labels, _ = batch
            # forward pass
            outputs = model(smiles_emb=smiles_emb, 
                            smiles_attn=smiles_attn, 
                            protein_emb=protein_emb,
                            protein_attn=protein_attn,
                            device=device,
                            gpu=gpu)
            
            loss = criterion(outputs, labels.float())
            val_loss += loss
            preds = outputs

            if args.binary_task:
                y_true.extend(labels.cpu().bool())
                y_pred.extend(preds.cpu())
            else:
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

    # added this:
    y_true = np.concatenate([np.array(y).flatten() for y in y_true])
    y_pred = np.concatenate([np.array(y).flatten() for y in y_pred])

    # calculate evaluation metrics
    if not args.binary_task:
        MSE = mean_squared_error(y_true, y_pred)
        val_loss /= len(valloader)
        R2 = r2_score(y_true, y_pred)
        CI = concordance_index(y_true, y_pred)
        logging.info('Val MSE: {:.4f}, Val Loss: {:.4f}, Val R2: {:.4f}, Val CI: {:.4f}'.format(MSE, val_loss, R2, CI))
        return val_loss, MSE
    else:
        y_true = np.array(y_true).astype(int)
        y_pred = np.array(y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        y_pred = np.round(y_pred).astype(int)
        acc = accuracy(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        val_loss /= len(valloader)
        logging.info('Val Accuracy: {:.4f},Val AUC: {:.4f},Val precision: {:.4f}, Val recall: {:.4f}, Val Loss: {:.4f}'.format(acc, auc_score, precision, recall, val_loss))
        return val_loss, acc


# Define the main function for training the model
def trainer(gpu, args, device, split):
    # Create a unique directory path for this iteration
    end_path, _ = os.path.splitext(os.path.basename(args.embed_path))
    log_dir = os.path.join(args.save_model_path, split, end_path)
    os.makedirs(log_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Create the log file path inside the directory
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

    # getting the dimension of the smiles
    #NOTE: This only takes 1D embeddings and ChemBERT is not a 1D embedding originally
    dim_size = list(pkl.load(open(args.embed_path, 'rb')).values())[0].shape[0]

    config = MM_TNConfig.from_dict({"s_hidden_size":dim_size, #NOTE: changed to accept other embeddings 1047, 724
        "p_hidden_size":1280,
        "hidden_size": 768,
        "max_seq_len":1276,
        "num_hidden_layers" : args.num_hidden_layers,
        "binary_task" : args.binary_task})

    logger.info(f"Loading model")
    model = MM_TN(config)
    
    if is_cuda(device):
        model = model.to(gpu)
        model = DDP(model, device_ids=[gpu])

    if os.path.exists(args.pretrained_model) and args.pretrained_model != "":
        logger.info(f"Loading model")
        try:
            state_dict = torch.load(args.pretrained_model)
            new_model_state_dict = model.state_dict()
            for key in new_model_state_dict.keys():
                if key in state_dict.keys():
                    try:
                        new_model_state_dict[key].copy_(state_dict[key])
                        logging.info("Updatete key: %s" % key)
                    except:
                        None
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


    # criterion = nn.MSELoss() #commented out hard coded loss func that doesnt allow for binar T/F values to change loss

    # added if/else statement to actually use dif loss funcs:
    if args.binary_task:
        logger.info(f'arg for binary_task: {args.binary_task}')
        criterion = nn.BCELoss()
        logger.info(f"using loss function: {criterion.__class__.__name__}")
    else:
        logger.info(f'arg for binary_task: {args.binary_task}')
        criterion = nn.MSELoss()
        logger.info(f"using loss function: {criterion.__class__.__name__}")
    #end of add
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # train the model
    logger.info(f"Start training")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.num_train_epochs):

        logger.info(f"Loading dataset to {device}:{gpu}")
        train_dataset = SMILESProteinDataset(
            data_path=os.path.join(args.train_dir, split, 'train_df.csv'),
            embed_dir = args.embed_path,
            train=True,
            device=device, 
            gpu=gpu,
            random_state = int(epoch),
            binary_task = args.binary_task,
            extraction_mode = False) 

        val_dataset = SMILESProteinDataset(
            data_path=os.path.join(args.train_dir, split, 'val_df.csv'),
            embed_dir = args.embed_path,
            train=False, 
            device=device, 
            gpu=gpu,
            random_state = int(epoch),
            binary_task = args.binary_task,
            extraction_mode = False)

        trainsampler = DistributedSampler(train_dataset, shuffle = False, num_replicas = args.world_size, rank = gpu, drop_last = True)
        valsampler = DistributedSampler(val_dataset, shuffle = False, num_replicas = args.world_size, rank = gpu, drop_last = True)


        logger.info(f"Loading dataloader")
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, sampler=trainsampler)
        valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, sampler=valsampler)
        

        start_time = time.time()
        logger.info(f"Training started for epoch: {epoch+1}")
        train_loss = train(args, model, trainloader, optimizer, criterion, device, gpu, epoch)
        logger.info(f"Training complete")

        del trainsampler
        del trainloader
        del train_dataset 

        logger.info(f"Val performance:")
        val_loss, val_mse = evaluate(args, model, valloader, criterion, device, gpu)
        logger.info(f"Evaluation complete")
        logger.info('-' * 80)
        logger.info(f'| Device id: {gpu} | End of epoch: {(epoch+1)} | '
                     f'Time taken: {(time.time()-start_time):5.2f}s |\n| Val CE loss: {val_loss:2.5f} | '
                     f'Val MSE {val_mse:2.5f} | Train Loss {train_loss:2.5f} |')
        logger.info('-' * 80)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if is_main_process():
                torch.save(model.state_dict(), os.path.join(log_dir, setting + '.pkl'))

    if args.world_size != -1:
        cleanup()


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
  
    # Loop through folds
    for split in os.listdir(args.train_dir):

        if not os.path.exists(os.path.join(args.save_model_path, split)):
            os.makedirs(os.path.join(args.save_model_path, split))

        try:
            if torch.cuda.is_available():
                mp.spawn(trainer, nprocs=args.world_size, args=(args, device, split))
            else:
                trainer(0, args, device, split)
        except Exception as e:
            print(e)
