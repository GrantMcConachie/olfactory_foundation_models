"""
Combination model of molecules and proteins for prediction. A linear regression
with a protein embeddings concatenated to a molecule embedding.

TODO:
 - scffold split molecules
"""

import os
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.model_selection import ShuffleSplit, GridSearchCV, KFold

import esm
import torch

import model.molecule_model as r


def generate_embeddings(dataset, mol_emb_type=('base', 'CATS')):
    """
    generates molecular and protein embeddings
    """
    # molecular embeddings
    mol_embs = r.generate_embeddings(dataset, mol_emb_type)

    # protein embeddings
    # check if saved
    parent_dir = os.path.dirname(dataset)
    save_path = os.path.join(parent_dir, 'featurized_proteins', 'prots.pkl')
    if os.path.isfile(save_path):
        return (mol_embs, pkl.load(open(save_path, 'rb')))

    # get all proteins
    prots = []
    for i, row in pd.read_csv(dataset).iterrows():
        prots.append((row['Protein sequence'][:5], row['Protein sequence']))

    prots = list(set(prots))

    # load esm model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # not prosmith
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # generated protein embeddings
    _, _, batch_tokens = batch_converter(prots)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # chunking into size 10 batches for cpu
    split_batch_tokens = torch.tensor_split(
        batch_tokens,
        len(batch_tokens) // 2,
        dim=0
    )
    split_batch_lens = torch.tensor_split(
        batch_lens,
        len(batch_tokens) // 2,
    )

    # Extract per-residue representations
    for (tok, lens) in zip(split_batch_tokens, split_batch_lens):
        with torch.no_grad():
            results = model(
                tok,
                repr_layers=[33],
                return_contacts=True
            )
        token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(
            token_representations[i, 1:tokens_len - 1].mean(0)
        )

    # putting representations into a dict and saving
    feat_dict = {}
    for p, rep in zip(prots, sequence_representations):
        feat_dict[p[1]] = rep

    pkl.dump(feat_dict, open(save_path, "wb"))

    return (mol_embs, feat_dict)


def save_splits(dataset, df, train_index, test_index, split_num, type):
    """
    Saves the splits as csv files
    """
    # creating save file
    save_fp = os.path.dirname(dataset) + f'/{type}_splits/'
    if not os.path.exists(save_fp):
        os.makedirs(save_fp)

    # saving dfs
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    train_df.to_csv(save_fp + f'train_{split_num}.csv', index=False)
    test_df.to_csv(save_fp + f'test_{split_num}.csv', index=False)


def cdhit_split(df, x, y, dataset):
    """
    splits data based on protein similarity
    """
    # paths of results from cdhit
    cluster_path = os.path.dirname(dataset) + '/cdhit/cd_out.clstr'
    prot_path = os.path.dirname(dataset) + '/cdhit/cd.fasta'

    # get all proteins and their labels
    seq = {}
    with open(prot_path, 'r') as f:
        for line in f:
            seq_id = line.split('>')[1].split('\n')[0]
            seq[seq_id] = next(f).split('\n')[0]

    # order proteins associated with their cluster
    ordered_prots = []
    with open(cluster_path, 'r') as f:
        for line in f:
            if line.startswith('>Cluster'):
                pass
            else:
                seq_id = line.split('>')[1].split('...')[0]
                ordered_prots.append(seq[seq_id])

    # put the indicies of the data in an order according to the clusters
    split_list = []
    for p in ordered_prots:
        split_list += list(df.index[df['Protein sequence'] == p])

    x_reorder = x[split_list]
    y_reorder = y[split_list]

    return x_reorder, y_reorder


def run_regression(dataset, mol_emb, prot_emb, regressor='r'):
    """
    Does 5 fold cross validation for different molecular embeddings with
    appended esm embeddings added
    """
    # init
    seed = 12345

    # getting data
    df = pd.read_csv(dataset)
    x, y = [], []
    for _, row in df.iterrows():
        mol_feat = mol_emb[row['SMILES']]
        prot_feat = prot_emb[row['Protein sequence']]
        combo = np.concatenate([mol_feat, prot_feat])
        x.append(combo)
        y.append(row['output'])

    # convert to numpy
    x = np.array(x)
    y = np.array(y)

    # parameters to sweep over
    param_grid = {
        'alpha': np.logspace(-10, 10, num=21)
    }

    # defining regressor
    if dataset == 'data/M2OR/pairs_ec50.csv':
        reg = LogisticRegression()
        param_grid = {
            'C': np.logspace(-10, 10, num=21)
        }
    elif regressor == 'r':
        reg = Ridge()
    elif regressor == 'l':
        reg = Lasso()

    # random splits of the data
    rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    rand_shuf_scores = []
    for i, (train_index, test_index) in enumerate(rs.split(y)):
        clf = GridSearchCV(
            reg,
            param_grid=param_grid,
        )
        clf.fit(x[train_index], y[train_index])
        best_model = clf.best_estimator_
        best_model.fit(x[train_index], y[train_index])
        save_splits(dataset, df, train_index, test_index, i, 'rand')

        # using mcc if m2or
        if dataset == 'data/M2OR/pairs_ec50.csv':
            preds = best_model.predict(x[test_index])
            score = matthews_corrcoef(y[test_index], preds)

        else:
            score = best_model.score(x[test_index], y[test_index])

        rand_shuf_scores.append(score)

    # cdhit splits of the data
    x, y = cdhit_split(df, x, y, dataset=dataset)
    cdhit_scores = []
    for i, (train_index, test_index) in enumerate(KFold(n_splits=5, shuffle=False).split(y)):
        clf = GridSearchCV(
            reg,
            param_grid=param_grid,
        )
        clf.fit(x[train_index], y[train_index])
        best_model = clf.best_estimator_
        best_model.fit(x[train_index], y[train_index])
        save_splits(dataset, df, train_index, test_index, i, 'cdhit')

        # using mcc if m2or
        if dataset == 'data/M2OR/pairs_ec50.csv':
            preds = best_model.predict(x[test_index])
            score_cd = matthews_corrcoef(y[test_index], preds)

        else:
            score_cd = best_model.score(x[test_index], y[test_index])

        cdhit_scores.append(score_cd)

    return (rand_shuf_scores, cdhit_scores)


def plot_results(emb_scores, dataset, ylabel='r2'):
    shuf_scores_NN = []
    shuf_scores_P = []
    cdhit_scores_NN = []
    cdhit_scores_P = []
    emb_names_NN = []
    emb_names_P = []

    for i in emb_scores:
        if i[0][0] == 'graph' or i[0][0] == 'trans':
            shuf_scores_NN.append(i[1])
            cdhit_scores_NN.append(i[2])
            emb_names_NN.append(i[0][1])
            print(f'{i[0][1]} shuf: mean {np.mean(i[1])} std: {np.std(i[1])}')
            print(f'{i[0][1]} cdhit: mean {np.mean(i[2])} std: {np.std(i[2])}')

        else:
            shuf_scores_P.append(i[1])
            cdhit_scores_P.append(i[2])
            emb_names_P.append(i[0][1])
            print(f'{i[0][1]} shuf: mean {np.mean(i[1])} std: {np.std(i[1])}')
            print(f'{i[0][1]} cdhit: mean {np.mean(i[2])} std: {np.std(i[2])}')

    # plotting
    for j, (nn, p) in enumerate([
        (shuf_scores_NN, shuf_scores_P),
        (cdhit_scores_NN, cdhit_scores_P)
    ]):
        fig, axs = plt.subplots(1, 2)

        # physicochemical descriptors
        axs[0].violinplot(
            np.array(p).T,
            showmeans=True,
            positions=np.arange(len(emb_names_P))
            )
        for i, row in enumerate(p):
            axs[0].scatter(
                np.ones_like(row) * i, row, color="#cc6666", alpha=0.7, s=5
            )
        axs[0].set_title('Physicochemical Descriptors')
        axs[0].set_ylabel(ylabel)
        axs[0].set_ylim([-3, 1])
        axs[0].set_xticks(range(len(emb_names_P)))
        axs[0].set_xticklabels(emb_names_P)

        # Neural network embeddings
        axs[1].violinplot(
            np.array(nn).T,
            showmeans=True,
            positions=np.arange(len(emb_names_NN))
        )
        for i, row in enumerate(nn):
            axs[1].scatter(
                np.ones_like(row) * i, row, color="#cc6666", alpha=0.3, s=5
            )
        # axs[1].violinplot(emb_scores_NN_scaf)
        axs[1].set_title('Neural Network Embeddings')
        axs[1].set_ylabel(ylabel)
        axs[1].set_ylim([-3, 1])
        axs[1].set_xticks(range(len(emb_names_NN)))
        axs[1].set_xticklabels(emb_names_NN)

        if j == 0:
            title = ' shuffle'
        else:
            title = ' cdhit'

        fig.suptitle(dataset + title)
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()


def main(dataset):
    emb_types_mol = r.get_embedding_types()
    for dataset in datasets:
        emb_scores = []
        for emb_type in tqdm(emb_types_mol):
            mol_emb, prot_emb = generate_embeddings(dataset, emb_type)
            mol_emb = r.clean_embeddings(mol_emb)

            # zscore mordred
            if emb_type == ('base', 'MordredDescriptors'):
                cleaned = np.array(list(mol_emb.values()))
                cleaned_z = zscore(cleaned)
                for i, (key, value) in enumerate(mol_emb.items()):
                    mol_emb[key] = cleaned_z[i]

                mol_emb = r.clean_embeddings(mol_emb)

            mol_emb = r.reduce_embs(mol_emb)  # idk about this
            shuf_scores, cdhit_scores = run_regression(
                dataset,
                mol_emb,
                prot_emb,
                regressor='r'
            )
            emb_scores.append((emb_type, shuf_scores, cdhit_scores))
        plot_results(emb_scores, dataset)
    plt.show()


if __name__ == '__main__':
    datasets = [
        # 'data/Davis/davis_z.csv',
        # 'data/HallemCarlson/hc_with_prot_seq_z.csv',
        # 'data/M2OR/pairs_ec50.csv',
        'data/CareyCarlson/CC_reformat_z.csv',
    ]
    main(datasets)
