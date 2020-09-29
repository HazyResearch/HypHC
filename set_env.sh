#!/bin/bash
source hyphc_env/bin/activate
export HHC_HOME=$(pwd)
export DATAPATH="$HHC_HOME/data"                # Path where to save the data files
export SAVEPATH="$HHC_HOME/embeddings"          # Path where to save the trained models 
