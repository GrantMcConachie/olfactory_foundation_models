"""
Generates protein and molecule embeddings

export HF_HOME=/projectnb/depaqlab/Grant/ProSmith_scc/downloads
export TORCH_HOME=/projectnb/depaqlab/Grant/ProSmith_scc/downloads
"""

import os
import shutil
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from scipy.stats import zscore

import torch

from molfeat.trans import MoleculeTransformer
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans.pretrained import PretrainedDGLTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer

from esm import FastaBatchedDataset, pretrained
from Bio import SeqIO


def get_mol_embedding_types():
    """
    Returns all embedding types in molfeat
    """
    rand = [('rand', 'rand')]
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

    return rand + ecfp + basic_fingerprints + transformers + gnns


def generate_mol_embeddings(dataset, emb_type):
    """
    Generates embeddings for all smiles in every dataset
    """
    # check if saved version
    emb_class = emb_type[0]
    emb_name = emb_type[1]
    parent_dir = os.path.dirname(os.path.dirname(dataset))
    save_path = os.path.join(parent_dir, 'embeddings', 'featurized_mols', f'{emb_name}.pkl')
    create_empty_path(os.path.dirname(save_path))

    # get all smiles
    smiles = list(pd.read_csv(dataset)['SMILES'])
    smiles = np.array(list(set(smiles)))

    # featurize
    if emb_class == 'base':
        featurizer = MoleculeTransformer(emb_name, dtype=np.float64)
        feats = featurizer(smiles)

    elif emb_class == 'ecfp':
        featurizer = FPVecTransformer(kind='ecfp', dtype=float)
        feats = featurizer(smiles)

    elif emb_class == 'trans':  # pretrained models
        featurizer = PretrainedHFTransformer(
            kind=emb_name,
            notation='smiles',
            dtype=float
        )
        feats = featurizer(smiles)

    elif emb_class == 'graph':
        featurizer = PretrainedDGLTransformer(
            kind=emb_name,
            dtype=float
        )
        feats = featurizer(smiles)

    elif emb_class == 'rand':
        feats = np.random.normal(size=(len(smiles), 300))

    # save
    feat_dict = {}
    for smi, feat in zip(smiles, feats):
        feat_dict[smi] = feat

    # zscore mordred
    if emb_type == ('base', 'MordredDescriptors'):
        embs = clean_mol_embeddings(feat_dict)
        cleaned = np.array(list(embs.values()))
        cleaned_z = zscore(cleaned)
        for i, (key, value) in enumerate(embs.items()):
            embs[key] = cleaned_z[i]

        embs = clean_mol_embeddings(embs)
        pkl.dump(embs, open(save_path, "wb"))
    
    else:
        pkl.dump(feat_dict, open(save_path, "wb"))


def clean_mol_embeddings(embs):
    """
    Removes NaNs from all the embeddigns if there is any. This is necessary
    for mordred descriptors.
    """
    values = np.array(list(embs.values()))
    nans = np.unique(np.where(np.isnan(values))[1])
    cleaned_embs = {}
    for key, value in embs.items():
        cleaned_embs[key] = np.delete(value, nans)

    return cleaned_embs


def calculate_protein_embeddings(dataset, prot_emb_no=1000):
    """
    Adapted from https://github.com/AlexanderKroll/ProSmith
    """
    # create folder
    parent_dir = os.path.dirname(os.path.dirname(dataset))
    outpath = os.path.join(parent_dir, 'embeddings', 'featurized_proteins')
    create_empty_path(outpath)

    df = pd.read_csv(dataset)
    all_sequences = list(set(list(df["Protein sequence"])))

    fasta_file = os.path.join(outpath, "all_sequences.fasta")
    create_fasta_file(all_sequences, fasta_file)

    model, alphabet = pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)

    output_dir = os.path.join(outpath, "temp")
    create_empty_path(output_dir)

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            # The model is trained on truncated sequences and passing longer ones in at
            # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
            toks = toks[:, :1022]

            out = model(toks, repr_layers=[33], return_contacts=False)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                output_file = os.path.join(output_dir, label + ".pt")

                result = {"label": label}
                result["representations"] = {
                    layer: t[i, 1 : len(strs[i]) + 1].clone()
                    for layer, t in representations.items()
                }

                torch.save(result, output_file)
    merge_protein_emb_files(output_dir, outpath, fasta_file, prot_emb_no)
    os.remove(fasta_file)


def merge_protein_emb_files(output_dir, outpath, fasta_file, prot_emb_no):
    """
    Adapted from https://github.com/AlexanderKroll/ProSmith
    """
    new_dict = {}

    version = 0
    fasta_sequences = SeqIO.parse(open(fasta_file),'fasta')

    for k, fasta in enumerate(fasta_sequences):

        if k %prot_emb_no == 0 and k > 0:
            torch.save(new_dict, os.path.join(outpath, "Protein_embeddings_V"+str(version)+".pt"))
            new_dict = {}
            version +=1

        name, sequence = fasta.id, str(fasta.seq)
        rep_dict = torch.load(os.path.join(output_dir, name +".pt"))
        new_dict[sequence] = rep_dict["representations"][33].numpy()

    torch.save(new_dict, os.path.join(outpath, "prots.pt"))

    shutil.rmtree(output_dir)


def create_fasta_file(sequences, filename):
    """
    Adapted from https://github.com/AlexanderKroll/ProSmith
    """
    ofile = open(filename, "w")
    for k, seq in enumerate(sequences):
        ofile.write(">" + str(k) + "\n" + seq[:1018]  + "\n")
    ofile.close()


def create_empty_path(path):
    try:
        os.makedirs(path)
    except:
        pass


if __name__ == '__main__':
    datasets = [
        # 'data/M2OR/raw/pairs_ec50.csv',
        # 'data/HC/raw/hc_with_prot_seq_z.csv',
        # 'data/CC/raw/CC_reformat_z.csv',
        'inference/data/inference.csv'
    ]

    emb_types = get_mol_embedding_types()

    for dataset in datasets:
        # generate protein embeddings
        calculate_protein_embeddings(dataset)

        # generating molecular embeddings
        for emb_type in tqdm(emb_types, desc=f'{dataset}'):
            generate_mol_embeddings(dataset, emb_type)
