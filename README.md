# Self Adaptive Visual Navigation

In recent years, there has been a lot of progress in visual navigation but it is not at the level where you can call it is  state-of-art. The one distinguishing framework that is attempting to provide a state-of-art navigation framework is the Ai2thor framework which also provides physics engine, object interaction and more. (In oreder to learn more about Ai2Thor Framework  and the original work on Self-Adaptive Visual Navigation please refer to https://ai2thor.allenai.org/ and https://github.com/allenai/savn respectively.

## Problem Statement
In the savn(https://github.com/allenai/savn) repo we based on, currenlty the agent is trained and tested in an offline Ai2thro environment, created by scraping images from a live environment and it is difficult to directly use the trained navigation models to a live Ai2thor environment. As part of our project, we aim to provide APIs so that users to create a live agent in a live Ai2thor simulator, where this agent uses pretrained models for visual navigation tasks. 


## Setup on local machine (Tested with MacBook Pro)

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


## Data for the Model
The `data` folder contains:

- `thor_glove` which contains the [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings for the navigation targets.

- `gcn` which contains the necessary data for the [Graph Convolutional Network (GCN)](https://arxiv.org/abs/1609.02907) in [Scene Priors](https://arxiv.org/abs/1810.06543), including the adjacency matrix.

** If you want to train the model, then you will need the offline-data which can obtained by downloading from this link - https://prior-datasets.s3.us-east-2.amazonaws.com/savn/data.tar.gz. Delete the current `data` folder, and extract the downloaded compressed data into project home folder. Training instructions ##

#### See a quick demo on jupyter notebook.
Run `jupyter notebook`
Open `online.ipynb`

#### if you want to test our API please execute the cells within the below notebook:

`test_import_online.ipynb`


## Notice
If you want to train the models, please see https://github.com/allenai/savn for full instruction


## Setup on docker 

Please refer the Dockerfile for creating the SAVN Docker Image. the 



## Setup on AWS with docker 

Choose the right image:
Deep Learning AMI (Ubuntu 16.04) Version 26.0 (ami-025ed45832b817a35)

This AMI comes with built support for nvidia drivers, docker environment.

Launch the EC2 instance with P2.2xlarge instance type with GPU support.

SSH into the launched EC2 instance by following instructions presented on Connect option.

Pull the docker image with command : docker pull sundaramx/savn-online:1.3

docker run --rm  -it --privileged -p 8888:8888 --hostname localhost sundaramx/savn-online:1.1

