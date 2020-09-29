# HypHC

This code is the official PyTorch implementation of the NeurIPS 2020 paper: "From Trees to Hyperbolic Embeddings and back: Hyperbolic Hierarchical Clustering". 

![alt text](https://github.com/HazyResearch/HypHC/blob/master/HypHC.gif)

<p align="center">
  <img width="300" height="300" src="https://github.com/HazyResearch/HypHC/blob/master/HypHC.gif">
</p>

## Installation

This code has been tested with python3.7.3. First, create a virtual environment (or conda environment) and install the dependencies:

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

### Code Examples

We provide examples of training commands for the zoo, iris and glass datasets. For instance, to train HypHC on zoo, run: 

```source examples/run_zoo.sh``` 

This will create an `embedding` directory and save embeddings in a `embedding/zoo/[unique_id]` where the unique id is based on the configuration parameters used to train the model.   

## Citation

If you find this code useful, please cite the following paper:
```
TODO(ines): add citation
```
