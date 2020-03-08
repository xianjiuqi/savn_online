## Setup

- Clone the repository with `git clone https://github.com/xianjiuqi/savn_online.git && cd savn_online`.

- Run `wget https://prior-datasets.s3.us-east-2.amazonaws.com/savn/pretrained_models.tar.gz`
Untar with
```bash
tar -xzf pretrained_models.tar.gz
```

## Run Pretrained Models in Ai2thor 2.2.0
#### See a quick demo on jupyter notebook. (Tested on Macbook Pro)
First create a conda environment. Assume miniconda3 is installed.
Run `conda create -n savn-online python=3.7`
In savn-onine folder, run `pip install -r requirements.txt`
Run `pip install jupyter`
Run `jupyter notebook`
Open `savn-online/online.ipynb` and run all the cells

#### Using our API
```bash
import online
"""
controller: Ai2thor controller
target: target object (str)
model_name: 'SAVN', 'NON_ADAPTIVE_A3C', or 'GCN'
model_path: path to pretrained model
glove_file_path: path to glove embedding
"""
episode = online.Episode(controller, target, model_name, model_path,glove_file_path)
episode.step() # let the controller step according to agent's action
episode.isolated_step(image) # return an action chosen by agent based on image

```


The `data` folder contains:

- `thor_glove` which contains the [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings for the navigation targets.
- `gcn` which contains the necessary data for the [Graph Convolutional Network (GCN)](https://arxiv.org/abs/1609.02907) in [Scene Priors](https://arxiv.org/abs/1810.06543), including the adjacency matrix.

## Notice
If you want to train the models, please see https://github.com/allenai/savn for full instruction






