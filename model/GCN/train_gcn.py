"""
Training scrpipt to run a GCN Baseline on all data
"""

import copy
import numpy as np
import pandas as pd
import deepchem as dc
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.model_selection import ShuffleSplit, KFold, StratifiedShuffleSplit

import torch
from torch_geometric.data import Data
from torch_geometric.nn.models import GCN
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool


def generate_graphs(smiles):
    """
    Generates graphs for all smiles
    """
    featurizer = dc.feat.MolGraphConvFeaturizer()
    graphs = featurizer.featurize(smiles)

    return graphs

    
def train_gcn(
        train_index,
        val_index,
        test_index,
        graphs,
        output,
        loss_fn,
        epochs=100,
        batch_size=32,
        lr=1e-4,
        weight_decay=1e-4
):
    """
    training loop for the gcn
    """
    device = torch.device(f'cuda' if torch.cuda.is_available() else "cpu")

    # convert graphs to pytroch geometric
    pyg_graphs = []
    for i, y in zip(graphs, output):
        x = torch.tensor(i.node_features, dtype=torch.float)
        edge_index = torch.tensor(i.edge_index, dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.float)).to(device)
        pyg_graphs.append(graph)

    # create dataloaders
    train_dataloader = DataLoader(
        [pyg_graphs[i] for i in train_index],
        batch_size=batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        [pyg_graphs[i] for i in val_index],
        batch_size=batch_size,
        shuffle=False
    )
    test_dataloader = DataLoader(
        [pyg_graphs[i] for i in test_index],
        batch_size=1,
        shuffle=False
    )
    
    # model
    model = GCN(
        in_channels=30,
        hidden_channels=128,
        num_layers=2,
        out_channels=1
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # training loop
    model.train()
    best_loss = 1e10
    for i in range(epochs):
        train_loss = []
        for dat in train_dataloader:
            optimizer.zero_grad()
            out = model(dat.x, dat.edge_index, batch=dat.batch)
            out = global_mean_pool(out, dat.batch).squeeze()
            loss = loss_fn(out, dat.y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss.append(loss)
        
        # evaluate validation performance
        with torch.no_grad():
            val_loss = []
            for dat in val_dataloader:
                out = model(dat.x, dat.edge_index, batch=dat.batch)
                out = global_mean_pool(out, dat.batch).squeeze()
                loss = loss_fn(out, dat.y.squeeze())
                val_loss.append(loss)

            if np.mean(val_loss) < best_loss:
                best_model = copy.deepcopy(model)
    
    # evaluate model on test data
    with torch.no_grad():
        preds = []
        for dat in test_dataloader:
            out = best_model(dat.x, dat.edge_index, batch=dat.batch)
            out = global_mean_pool(out, dat.batch).squeeze()
            preds.append(out.cpu().item())

    return preds


def run_cross_val(dataset):
    """
    Runs a 5 fold cross validation using the GCN model
    """
    # init
    seed = 12345
    ss = KFold(n_splits=5, shuffle=False)
    rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    df = pd.read_csv(dataset)
    proteins = df['Protein sequence'].unique()

    # loop through individual proteins
    scores_per_protein = []
    for protein in tqdm(proteins, desc='protein'):
        # get smiles and output
        smiles = df['SMILES'][df['Protein sequence'] == protein]
        output = np.array(df['output'][df['Protein sequence'] == protein])

        # change loss for different models
        if dataset == 'data/M2OR/raw/pairs_ec50.csv':
            loss_fn = torch.nn.BCEWithLogitsLoss()

            # ignore proteins with < 50 values
            if len(smiles) < 50:
                continue

            # ignore output with only 1 positive example
            if len(np.where(output == 1)[0]) < 2 or len(np.where(output == 0)[0]) < 2:
                continue

            # stratify splits
            rs = StratifiedShuffleSplit(n_splits=5, random_state=seed)
            splits = rs.split(output, output)
        
        elif dataset == 'data/CC/raw/CC_reformat_z.csv':
            # remove 'N' because it cannot be a graph
            output = output[smiles != 'N']
            smiles = smiles[smiles != 'N']
            loss_fn = torch.nn.MSELoss()
            splits = rs.split(output)

        else:
            loss_fn = torch.nn.MSELoss()
            splits = rs.split(output)

        # generate graphs
        graphs = generate_graphs(smiles)

        # 5-fold cross validaiton
        random_shuf_scores = []
        for i, (train_index, test_index) in enumerate(splits):
            # create a small validation dataset
            val_index = train_index[:round(len(train_index) * 0.15)]
            mask = ~np.isin(train_index, val_index)
            train_index = train_index[mask]

            preds = train_gcn(
                train_index,
                val_index,
                test_index,
                graphs,
                output,
                loss_fn
            )

            # get scores
            if dataset == 'data/M2OR/raw/pairs_ec50.csv':
                score = matthews_corrcoef(output[test_index], (np.array(preds)>0).astype(int))
            else:
                score = r2_score(output[test_index], preds)
            random_shuf_scores.append(score)
    
        # append protein score
        print(f"5-fold scores: {random_shuf_scores}")
        scores_per_protein.append(random_shuf_scores)

    # print results
    all_scores = [i for sublist in scores_per_protein for i in sublist]
    print(f'mean score {dataset}')
    print(np.mean(all_scores))
    print('std score')
    print(np.std(all_scores))


def main(datasets):
    for dataset in datasets:
        run_cross_val(dataset)


if __name__ == '__main__':
    datasets = [
        'data/M2OR/raw/pairs_ec50.csv',
        # 'data/HC/raw/hc_with_prot_seq_z.csv',
        # 'data/CC/raw/CC_reformat_z.csv'
    ]
    main(datasets)
