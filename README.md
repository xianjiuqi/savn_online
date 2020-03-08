# Self Adaptive Visual Navigation

In recent years, there has been a lot of progress in visual navigation but it is not at the level where you can call it is  state-of-art. The one distinguishing framework that is attempting to provide a state-of-art navigation framework is the Ai2thor framework which also provides physics engine, object interaction and more. (In oreder to learn more about Ai2Thor Framework  and the original work on Self-Adaptive Visual Navigation please refer to https://ai2thor.allenai.org/ and https://github.com/allenai/savn respectively.

## Problem Statement
In the SAVN repo, currenlty the agent is trained and tested in an offline Ai2thro environment, created by scraping images from a live environment. And it is difficult to directly use the trained navigation models to a live Ai2thor environment. As part of our project, we aim to provide APIs for user to create a live agent in a live Ai2thor simulator, where this agent uses pretrained models for visual navigation tasks. 



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
#### See a quick demo on jupyter notebook. (Tested on Macbook Pro/ AWS / Docker)
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






