# Hyperbolic Hierarchical Clustering (HypHC)

This code is the official PyTorch implementation of the NeurIPS 2020 paper: 
> **From Trees to Continuous Embeddings and Back: Hyperbolic Hierarchical Clustering**\
> Ines Chami, Albert Gu, Vaggos Chatziafratis and Christopher Ré\
> Stanford University\
> Paper: TODO(ines): add link

<p align="center">
  <img width="400" height="400" src="https://github.com/HazyResearch/HypHC/blob/master/HypHC.gif">
</p>

> **Abstract.** Similarity-based Hierarchical Clustering (HC) is a classical unsupervised machine learning algorithm that has traditionally been solved with heuristic algorithms like Average-Linkage. Recently, Dasgupta reframed HC as a discrete optimization problem by introducing a global cost function measuring the quality of a given tree. In this work, we provide the first continuous relaxation of Dasgupta's discrete optimization problem with provable quality guarantees. The key idea of our method, HypHC, is showing a direct correspondence from discrete trees to continuous representations (via the hyperbolic embeddings of their leaf nodes) and back (via a decoding algorithm that maps leaf embeddings to a dendrogram), allowing us to search the space of discrete binary trees with continuous optimization. Building on analogies between trees and hyperbolic space, we derive a continuous analogue for the notion of lowest common ancestor, which leads to a continuous relaxation of Dasgupta's discrete objective. We can show that after decoding, the global minimizer of our continuous relaxation yields a discrete tree with a (1+epsilon)-factor approximation for Dasgupta's optimal tree, where epsilon can be made arbitrarily small and controls optimization challenges. We experimentally evaluate HypHC on a variety of HC benchmarks and find that even approximate solutions found with gradient descent have superior clustering quality than agglomerative heuristics or other gradient based algorithms.Finally, we highlight the flexibility of HypHC using end-to-end training in a downstream classification task.


## Installation

This code has been tested with python3.7. First, create a virtual environment (or conda environment) and install the dependencies:

```python3 -m venv hyphc_env```

```source hyphc_env/bin/activate```

```pip install -r requirements.txt``` 

Then install the ```mst``` and ```unionfind``` packages which are used to decode embeddings into trees and compute the discrete Dasgupta Cost efficiently: 

```cd mst; python setup.py build_ext --inplace```

```cd unionfind; python setup.py build_ext --inplace```

## Datasets

```source download_data.sh```

This will download the zoo, iris and glass datasets from the UCI machine learning repository. Please refer to the paper for the download links of the other datasets used in the paper. 

## Code Usage

### Train script

To use the code, first set environment variables in each shell session:

```source set_env.sh```

To train the HypHC mode, use the train script:
```
python train.py
    optional arguments:
      -h, --help            show this help message and exit
      --seed SEED
      --epochs EPOCHS
      --batch_size BATCH_SIZE
      --learning_rate LEARNING_RATE
      --eval_every EVAL_EVERY
      --patience PATIENCE
      --optimizer OPTIMIZER
      --save SAVE
      --fast_decoding FAST_DECODING
      --num_samples NUM_SAMPLES
      --dtype DTYPE
      --rank RANK
      --temperature TEMPERATURE
      --margin MARGIN
      --init_size INIT_SIZE
      --anneal_every ANNEAL_EVERY
      --anneal_factor ANNEAL_FACTOR
      --max_scale MAX_SCALE
      --dataset DATASET
``` 

### Examples

We provide examples of training commands for the zoo, iris and glass datasets. For instance, to train HypHC on zoo, run: 

```source examples/run_zoo.sh``` 

This will create an `embedding` directory and save embeddings in a `embedding/zoo/[unique_id]` where the unique id is based on the configuration parameters used to train the model.   

## Citation

If you find this code useful, please cite the following paper:
```
TODO(ines): add citation
```
