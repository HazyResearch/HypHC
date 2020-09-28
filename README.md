# HypHC

This code is the official PyTorch implementation of the NeurIPS 2020 paper: "From Trees to Hyperbolic Embeddings and back: Hyperbolic Hierarchical Clustering". 

#### Download data

```source download_data.sh```

#### Installation

python>=3.5

```pip install -r requirements.txt``` 

Install ```mst``` and ```unionfind``` packages. 

```python setup.py build_ext --inplace```

#### Usage

First, set environment variables:

```source set_env.sh```

Then, train HypHC:
```
python train.py
    optional arguments:
      -h, --help            show this help message and exit
      --seed SEED
      --epochs EPOCHS
      --batch_size BATCH_SIZE
      --learning_rate LEARNING_RATE
      --eval_every EVAL_EVERY
      --optimizer OPTIMIZER
      --save SAVE
      --dtype DTYPE
      --rank RANK
      --temperature TEMPERATURE
      --margin MARGIN
      --init_size INIT_SIZE
      --curvature CURVATURE
      --anneal_every ANNEAL_EVERY
      --anneal_factor ANNEAL_FACTOR
      --dataset DATASET
``` 

