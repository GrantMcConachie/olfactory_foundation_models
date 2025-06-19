"""
Benchmarks all featurizers on molfeat against OR datasets of interest.
"""

import os
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold, StratifiedShuffleSplit

import deepchem as dc


def get_mol_embedding_types():
    """
    Returns all embedding types in molfeat
    """
    transformers = [
        ('trans', 'Roberta-Zinc480M-102M'),
        ('trans', 'GPT2-Zinc480M-87M'),
        ('trans', 'ChemGPT-19M'),
        ('trans', 'MolT5'),
        ('trans', 'ChemBERTa-77M-MTR'),
    ]
    gnns = [
        ('graph', 'gin_supervised_infomax'),
        ('graph', 'gin_supervised_edgepred'),
        ('graph', 'gin_supervised_contextpred')
    ]
    basic_fingerprints = [
        ('base', 'CATS'),
        ('base', 'MordredDescriptors'),
        ('base', 'Pharmacophore2D'),
        ('base', 'RDKitDescriptors2D'),
        ('base', 'ScaffoldKeyCalculator')
    ]
    ecfp = [('ecfp', 'ecfp')]

    return basic_fingerprints + ecfp + transformers + gnns


def get_mol_embedding(dataset, emb_type):
    """
    Loads embedding 'data/CC/raw/CC_reformat_z.csv'
    """
    # embedding path
    dir = os.path.dirname(os.path.dirname(dataset))
    dir = os.path.join(dir, 'embeddings', 'featurized_mols', f'{emb_type[1]}.pkl')

    return pkl.load(open(dir, 'rb'))


def reduce_embs(emb_dict):
    """
    Does PCA on embeddings and reduces the dimensionality to explain 99% of the
    variance
    """
    embs = np.array(list(emb_dict.values()))
    reduced = PCA(n_components=0.99, svd_solver='full').fit_transform(embs)
    reduced_embs = {}
    for i, (key, _) in enumerate(emb_dict.items()):
        reduced_embs[key] = reduced[i]

    return reduced_embs


def run_lr(dataset, embs, regressor='r'):
    """
    Runs a 5-fold linear regression for a given embedding and dataset
    """
    # init
    seed = 12345
    scaffoldsplitter = dc.splits.ScaffoldSplitter()
    ss = KFold(n_splits=5, shuffle=False)
    rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    df = pd.read_csv(dataset)
    proteins = df['Protein sequence'].unique()

    # loop through unique proteins
    scores_per_protein = []
    for protein in proteins:
        smiles = df['SMILES'][df['Protein sequence'] == protein]
        output = np.array(df['output'][df['Protein sequence'] == protein])
        embeddings = np.array([embs[i] for i in smiles])

        # parameters to sweep over
        param_grid = {
            'alpha': np.logspace(-10, 10, num=21)
        }

        # defining regressor
        if dataset == 'data/M2OR/pairs_ec50.csv':  # binary data
            reg = LogisticRegression()
            param_grid = {
                'C': np.logspace(-10, 10, num=21)
            }

            # ignore proteins with < 50 values
            if len(smiles) < 50:
                continue

            # ignore output with only 1 positive example
            if len(np.where(output == 1)[0]) < 2 or len(np.where(output == 0)[0]) < 2:
                continue

            # have to have stratified splitting
            rs = StratifiedShuffleSplit(n_splits=5, random_state=seed)
            splits = rs.split(embeddings, output)

        elif regressor == 'r':
            reg = Ridge()
            splits = rs.split(output)

        elif regressor == 'l':
            reg = Lasso()
            splits = rs.split(output)

        # 5-fold cross val for random shuffling
        random_shuf_scores = []
        for i, (train_index, test_index) in enumerate(splits):
            clf = GridSearchCV(
                reg,
                param_grid=param_grid,
                n_jobs=-1
            )
            clf.fit(embeddings[train_index], output[train_index])
            best_model = clf.best_estimator_
            best_model.fit(embeddings[train_index], output[train_index])

            # change scoring based on dataset
            if dataset == 'data/M2OR/pairs_ec50.csv':
                preds = best_model.predict(embeddings[test_index])
                score = matthews_corrcoef(output[test_index], preds)
                random_shuf_scores.append(score)

            else:
                r2 = best_model.score(embeddings[test_index], output[test_index])
                random_shuf_scores.append(r2)

        # 5-fold cross val for scaffold splitting
        if dataset == 'data/M2OR/pairs_ec50.csv':
            # no scaffold for M2OR
            scaf_shuf_scores = np.zeros((5,))

        else:
            scaf_shuf_scores = []
            dataset_ = dc.data.NumpyDataset(
                X=embeddings,
                y=output,
                ids=smiles
            )
            scaf_out = np.array([
                output[x]
                for xs in scaffoldsplitter.generate_scaffolds(dataset_)
                for x in xs
            ])
            scaf_emb = np.array([
                embeddings[x]
                for xs in scaffoldsplitter.generate_scaffolds(dataset_)
                for x in xs
            ])

            for i, (train_index, test_index) in enumerate(ss.split(scaf_out)):
                clf = GridSearchCV(
                    reg,
                    param_grid=param_grid,
                    n_jobs=-1
                )
                clf.fit(scaf_emb[train_index], scaf_out[train_index])
                best_model = clf.best_estimator_
                best_model.fit(scaf_emb[train_index], scaf_out[train_index])
                r2 = best_model.score(scaf_emb[test_index], scaf_out[test_index])
                scaf_shuf_scores.append(r2)

        scores_per_protein.append(
            (
                protein,
                random_shuf_scores,
                scaf_shuf_scores
            )
        )

    return scores_per_protein


def plot_results(results, dataset, regressor, ylabel='r2/mcc'):
    """
    plots embedding scores on a given dataset
    """
    # getting average score over all proteins
    emb_names_NN = []
    emb_scores_NN_shuf = []
    emb_scores_NN_scaf = []
    emb_names_P = []
    emb_scores_P_shuf = []
    emb_scores_P_scaf = []

    print(dataset, f'regressor: {regressor}')
    for result in results:
        shuf_protein_scores = np.array([i[1] for i in result[1]]).flatten()
        scaf_protein_scores = np.array([i[2] for i in result[1]]).flatten()
        if result[0][0] == 'graph' or result[0][0] == 'trans':
            emb_names_NN.append(result[0][1])
            emb_scores_NN_shuf.append(shuf_protein_scores)
            emb_scores_NN_scaf.append(scaf_protein_scores)

        else:
            emb_names_P.append(result[0][1])
            emb_scores_P_shuf.append(shuf_protein_scores)
            emb_scores_P_scaf.append(scaf_protein_scores)

        # ouput to terminal
        print(result[0][1])
        print('shuf mean:', np.mean(shuf_protein_scores))
        print('shuf std:', np.std(shuf_protein_scores))
        print('scaf mean:', np.mean(scaf_protein_scores))
        print('scaf std:', np.std(scaf_protein_scores), '\n')

    # plot for scaffold and random shuffled
    for j, (nn, p) in enumerate([
        (emb_scores_NN_scaf, emb_scores_P_scaf),
        (emb_scores_NN_shuf, emb_scores_P_shuf)
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
                np.ones_like(row) * i, row, color="#cc6666", alpha=0.3, s=5
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
        axs[1].set_title('Neural Network Embeddings')
        axs[1].set_ylabel(ylabel)
        axs[1].set_ylim([-3, 1])
        axs[1].set_xticks(range(len(emb_names_NN)))
        axs[1].set_xticklabels(emb_names_NN)

        if j == 0:
            title = " scaffold split"
        else:
            title = " shuffle split"

        fig.suptitle(dataset + title)
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()


def tabulate(emb_scores, dataset, regressor):
    """
    Saves data to a csv
    """
    # create empty dataframe and a place for it to go
    path = os.path.dirname(dataset).replace("data/", "results/")
    path += "/molecule_emb_only.csv"
    df = pd.DataFrame(
        columns=[
            'embedding',
            'shuf_mean',
            'scaf_mean',
            'shuf1',
            'shuf2',
            'shuf3',
            'shuf4',
            'shuf5',
            'scaf1',
            'scaf2',
            'scaf3',
            'scaf4',
            'scaf5',
            'receptor',
            'regressor'
        ]
    )

    # loop through scores
    for score in emb_scores:
        emb = score[0][1]
        for prot in score[1]:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame({
                        'embedding': [emb],
                        'shuf_mean': [np.mean(prot[1])],
                        'scaf_mean': [np.mean(prot[2])],
                        'shuf1': [prot[1][0]],
                        'shuf2': [prot[1][1]],
                        'shuf3': [prot[1][2]],
                        'shuf4': [prot[1][3]],
                        'shuf5': [prot[1][4]],
                        'scaf1': [prot[2][0]],
                        'scaf2': [prot[2][1]],
                        'scaf3': [prot[2][2]],
                        'scaf4': [prot[2][3]],
                        'scaf5': [prot[2][4]],
                        'receptor': [prot[0]],
                        'regressor': regressor
                    })
                ],
                ignore_index=True
            )

    # save
    df.to_csv(path, index=False)


def main(datasets, regressor='r'):
    emb_types = get_mol_embedding_types()
    for dataset in datasets:
        emb_scores = []
        for emb_type in tqdm(emb_types, desc=f'{dataset}'):
            embs = get_mol_embedding(dataset, emb_type)
            embs = reduce_embs(embs)
            scores = run_lr(dataset, embs, regressor=regressor)
            emb_scores.append((emb_type, scores))

        plot_results(emb_scores, dataset, regressor)
        tabulate(emb_scores, dataset, regressor)

    plt.show()


if __name__ == '__main__':
    datasets = [
        # 'data/M2OR/raw/pairs_ec50.csv',
        # 'data/HC/raw/hc_with_prot_seq_z.csv',
        'data/CC/raw/CC_reformat_z.csv'
    ]
    main(datasets)
