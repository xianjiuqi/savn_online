## Setup

- Clone the repository with `git clone https://github.com/xianjiuqi/savn_online.git && cd savn_online`.

- Run `wget https://prior-datasets.s3.us-east-2.amazonaws.com/savn/pretrained_models.tar.gz`
Untar with
```bash
tar -xzf pretrained_models.tar.gz
```
- Create a conda environment. Assume miniconda3 is installed.
- Run `conda create -n savn-online python=3.7`
- Run `conda activate savn-online`
- In savn-onine folder, run `pip install -r requirements.txt`

## Run Pretrained Models in Ai2thor 2.2.0
#### See a quick demo on jupyter notebook. (Tested on Macbook Pro)
Run `jupyter notebook`
Open `online.ipynb`

#### Using our API
See a tutorial 
Open `test_import_online.ipynb`


The `data` folder contains:

- `thor_glove` which contains the [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings for the navigation targets.
- `gcn` which contains the necessary data for the [Graph Convolutional Network (GCN)](https://arxiv.org/abs/1609.02907) in [Scene Priors](https://arxiv.org/abs/1810.06543), including the adjacency matrix.

## Notice
If you want to train the models, please see https://github.com/allenai/savn for full instruction






