# ProSmith
This is a cloned version of the ProSmith repo that was refined to include setting up ProSmith on the SCC. There are a few additional scripts for properly formatting data and submitting batch jobs on the scc. 

[Original ProSmith repo](https://github.com/AlexanderKroll/ProSmith)\
[Methods paper describing the ProSmith architecture](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012100#sec002)


## Downloading data folder
The original repo had their code and data available for [download from Zenodo](https://doi.org/10.5281/zenodo.10986299).
Afterwards, this repository should have the following strcuture:

    ├── code
    ├── data   
    ├── LICENSE.md     
    └── README.md

The relevant data for M2OR is in my directory: /projectnb/depaqlab/emily/m2or/pairs.csv \
It can also be downloaded from the [M2OR website](https://m2or.chemsensim.fr/)

## Install
```
conda env create -f environment.yml
conda activate prosmith
pip install -r requirements.txt
```
If you run into issues with torch,  make sure you didn't install the cpu only version:
```
conda remove pytorch torchvision torchaudio cpuonly -n prosmith
conda install pytorch=2.0.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

```


## How to train and evaluate a model
### (a) Data preparation
First, you need a training, validation, and test set as csv files. Every csv file should be comma-separated and have the following columns:\
-Protein sequence: contains the protein amino acid sequences of proteins\
-SMILES: contains SMILES strings for small molecules\
-output: The value of the target variable for the protein-small molecule pair of this row

For an example of such csv-files, please have a look at the files in following folder of this repositiory: "data/training_data/ESP/train_val".
### (a.1) CD-HIT
 [CD-HIT user guide](https://github.com/weizhongli/cdhit/wiki/3.-User's-Guide#cd-hit)
To use cd-hit to cluster similar sequences together, you must first install cd-hit in your conda env:
```python
conda deactivate
conda env --name cd-hit
conda activate cd-hit
conda install -c bioconda cd-hit
```
 cd-hit needs a .fasta file with all sequences to cluster, heres an example of how to generate that from you data csv:
```python
humanpairs = pd.read_csv('/projectnb/depaqlab/emily/m2or/human/humanpairs.csv')


with open('/projectnb/depaqlab/emily/m2or/intermediatefiles/human_cd_sequences.fasta', 'w') as f:
    for i, row in humanpairs.iterrows():
        f.write(f'>seq{i}\n{row["Sequence"]}\n')
```
 in the terminal where you have your conda env that has cd-hit installed run the following:
 ```python
cd-hit -i /projectnb/depaqlab/emily/m2or/intermediatefiles/human_cd_sequences.fasta -o /projectnb/depaqlab/emily/m2or/intermediatefiles/cd-hit/test/clustered_human_seq -c 0.8 -n 5 -M 8000 -d 0 -T 8
```
This gives you 2 output files: one with the sequences of all the reference sequences and another .clstr file that has the clusters with the reference seq denoted with a '*' and the sequences below it being part of that cluster w/ their associated % similarity. Sequence #s come from the index in the input dataframe and will be used to help split the data by clusters.

 ```python
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

humanpairs = pd.read_csv('/projectnb/depaqlab/emily/m2or/human/humanpairs.csv')
clusters = {}
with open('/projectnb/depaqlab/emily/m2or/intermediatefiles/cd-hit/test/clustered_human_seq.clstr') as f:
    cluster_id = -1
    for line in f:
        if line.startswith('>Cluster'):
            cluster_id += 1
            clusters[cluster_id] = []
        else:
            seq_id = line.split('>')[1].split('...')[0]
            clusters[cluster_id].append(seq_id)

# splitting up clusters:
## - note: this splitting doesn't take number of sequences into account so there will be size differences in test/val sets even though they should both be 10% of the data
cluster_ids = list(clusters.keys())
random.shuffle(cluster_ids)

train_ratio = 0.8
val_test_ratio = 0.1

train_clusters = cluster_ids[:int(train_ratio * len(cluster_ids))]
val_clusters = cluster_ids[int(train_ratio * len(cluster_ids)):int((train_ratio + val_test_ratio) * len(cluster_ids))]
test_clusters = cluster_ids[int((train_ratio + val_test_ratio) * len(cluster_ids)):]

train_set = []
val_set = []
test_set = []

for cluster_id in train_clusters:
    train_set.extend(clusters[cluster_id])

for cluster_id in val_clusters:
    val_set.extend(clusters[cluster_id])

for cluster_id in test_clusters:
    test_set.extend(clusters[cluster_id])

# map sequences back into the df:
humanpairs['Set'] = 'None'
for i in train_set:
    idx = int(i[3:])
    humanpairs.at[idx, 'Set'] = 'Train'

for i in val_set:
    idx = int(i[3:])
    humanpairs.at[idx, 'Set'] = 'Validation'

for i in test_set:
    idx = int(i[3:])
    humanpairs.at[idx, 'Set'] = 'Test'


traindf = humanpairs[humanpairs['Set'] == 'Train']
valdf = humanpairs[humanpairs['Set'] == 'Validation']
testdf = humanpairs[humanpairs['Set'] == 'Test']

# have to rename columns so prosmith properly retrieves data
traindf = traindf.rename(columns={"Sequence": "Protein sequence", "Responsive": "output"}) 
valdf = valdf.rename(columns={"Sequence": "Protein sequence", "Responsive": "output"})
testdf = testdf.rename(columns={"Sequence": "Protein sequence", "Responsive": "output"})

traindf.to_csv('/projectnb/depaqlab/emily/m2or/human/withcd/trainvaltest/train_df.csv', index=False)
valdf.to_csv('/projectnb/depaqlab/emily/m2or/human/withcd/trainvaltest/val_df.csv', index=False)
testdf.to_csv('/projectnb/depaqlab/emily/m2or/human/withcd/trainvaltest/test_df.csv', index=False)

```


### (b) Calculating input representations for proteins and small molecules:
Before running the below command set two environment variables so the saved models do not save into your home directory. Set some folder where things can get downloaded into.

```
export HF_HOME=/projectnb/depaqlab/Grant/ProSmith_scc/downloads
export TORCH_HOME=/projectnb/depaqlab/Grant/ProSmith_scc/downloads

```

Before you can start training ProSmith, ESM-1b embeddings and ChemBERTa2 embeddings need to be calculated for all protein sequences and SMILES strings in your repository, respectively. For example, for the M2OR dataset this can be done by executing the following command:

```python
conda deactivate
conda activate prosmith
python /projectnb/depaqlab/Grant/ProSmith_scc/code/preprocessing/preprocessing.py --train_val_path /projectnb/depaqlab/Grant/ProSmith_scc/door/train_val_test/ --outpath /projectnb/depaqlab/Grant/ProSmith_scc/door/embeddings --smiles_emb_no 2000 --prot_emb_no 2000
```
-"train_val_path": specifies, where all training and validation files are stored (with all protein sequences and SMILES strings)\
-"outpath": specifies, where the calculated ESM-1b and ChemBERTa2 embeddings will be stored\
-"smiles_emb_no" & "prot_emb_no": specify how many ESM-1b and ChemBERTa2 embeddings are stored in 1 file. 2000 for each variable worked well on a PC with 16GB RAM. Increased numbers might lead to better performance during model training.


### (c) Training the ProSmith Transformer Network:
To train the ProSmith Transformer Network (code is an example to train for human data that has been split with cd-hit for the M2OR task):

```python
python /projectnb/depaqlab/emily/prosmith/github/ProSmith-main/code/training/training.py --train_dir /projectnb/depaqlab/emily/m2or/human/withcd/trainvaltest/train_df.csv --val_dir /projectnb/depaqlab/emily/m2or/human/withcd/trainvaltest/val_df.csv --embed_path /projectnb/depaqlab/emily/m2or/human/withcd/embeddings --save_model_path /projectnb/depaqlab/emily/m2or/human/withcd/saved_model --batch_size 12 --binary_task True --learning_rate 1e-5 --num_train_epochs 50 --start_epoch 0 --warmup_proportion 0.1 --num_hidden_layers 6 --port 12557 --log_name humancdm2ortrain --pretrained_model ''
```
To train the model on the SCC GPUs, I used the batch script that can be found in /projectnb/depaqlab/emily/prosmith/github/ProSmith-main/humantrain_cd_send.sh. This has also been uploaded to the folder "batch_ex"
This model will train for num_train_epochs=50 epochs and it will store the best model (i.e. with the best performance on the validation set) in "save_model_path". Therefore, after each epoch model performance is evaluated. 
"binary_task" is set to True, because the ESP prediction task is a binary classification task. The variable has to be set to False for regression tasks.


### (d) Training the Gradient Boosting Models:
To train gradient boosting models for the ESP task, execute the following command:

```python
python /projectnb/depaqlab/emily/prosmith/github/ProSmith-main/code/training/training_GB.py --train_dir /projectnb/depaqlab/emily/m2or/human/withcd/trainvaltest/train_df.csv --val_dir /projectnb/depaqlab/emily/m2or/human/withcd/trainvaltest/val_df.csv --test_dir /projectnb/depaqlab/emily/m2or/human/withcd/trainvaltest/test_df.csv --pretrained_model /projectnb/depaqlab/emily/m2or/human/withcd/saved_model/humancdm2ortrain_4gpus_bs48_1e-05_layers6.txt.pkl --embed_path /projectnb/depaqlab/emily/m2or/human/withcd/embeddings --save_pred_path /projectnb/depaqlab/emily/m2or/human/withcd/saved_predictions --num_hidden_layers 6 --num_iter 500 --log_name humanwithcdGB --binary_task True		    
```
The batch script used to submit this job can be found in /projectnb/depaqlab/emily/prosmith/github/ProSmith-main/human_cd_trainGB_send.sh and is also in the batch example folder.
The final predictions of the ProSmith model will be saved in "save_pred_path". They might differ from the original order in the csv file, but there is an additional file, containing the original indices. You can map the predictions to the csv file using the following python code:


```python
import numpy as np
import pandas as pd
from os.path import join

#loading predictions and test set:
y_pred = np.load(join(path_to_repository, "data","training_data", "ESP", "saved_predictions", "y_test_pred.npy"))
y_pred_ind = np.load(join(path_to_repository, "data","training_data", "ESP", "saved_predictions", "test_indices.npy"))
test_df = pd.read_csv(join(path_to_repository, "data, "training_data", ESP", "train_val", "ESP_test_df.csv"))

#Mapping predictions to test set:
test_df["y_pred"] = np.nan
for k, ind in enumerate(y_pred_ind):
    test_df["y_pred"][ind] = y_pred[k]
```


## Requirements for running the code in this GitHub repository
There may be an issue with setting up pytorch using the requirements and yaml file in this repo, I had to uninstall the cpu-only version of pytorch and install the correct version along with the correct cuda toolkit. 
The code was implemented and tested on Linux with the following packages and versions
- python 3.8.3
- pandas 1.3.0
- torch 2.0.0+cu11.7
- numpy 1.22.4
- Bio 1.79
- transformers 4.27.2
- logging 0.5.1.2
- sklearn 1.2.2
- lifelines 0.27.7
- xgboost 0.90
- hyperopt 0.2.5
- json 2.0.9


## Misc. Notes/Future Directions 
M2OR - the french group that used this data for their GNN found that using only the ec50 (highest quality data) yielded better results, this is a small portion of the available data but if the primary screening data is truly noisy then it would make sense to throw it out.
DoOR - ProSmith needs the AA sequences in order to run and those are not included in the DoOR data, I compiled a tiny csv with the top listed AA seq from UniProt but there are different protein AA sequence isoforms due to alternative splicing which may mean that the small molecule-receptor pairing may not be completely accurate.
 Future directions: tweaking loss functions, weighting data samples, changing how embeddings are extracted, more hyperparameter tuning across the board but especially in the gradient boosting models, combining data from DoOR with M2OR and other sources to have a larger cross-species dataset
