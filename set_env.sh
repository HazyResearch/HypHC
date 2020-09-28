#!/bin/bash
export HHC_HOME=$(pwd)
export PATH="$HHC_HOME:$PATH"
export PYTHONPATH="$HHC_HOME:$PYTHONPATH"
export DATAPATH="$HHC_HOME/data"
export SAVEPATH="$HHC_HOME/embeddings"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
source activate hgcn
