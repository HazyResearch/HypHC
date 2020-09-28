#!/bin/bash
cd $HHC_HOME
mkdir data
for dataset in zoo iris glass; do
  mkdir data/$dataset
  wget -P data/$dataset https://archive.ics.uci.edu/ml/machine-learning-databases/$dataset/$dataset.data
  wget -P data/$dataset https://archive.ics.uci.edu/ml/machine-learning-databases/$dataset/$dataset.names
done
