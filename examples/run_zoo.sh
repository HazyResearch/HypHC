#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $DIR
dataset="zoo"
source set_env.sh
python train.py --dataset $dataset \
                --epochs 201 \
                --batch_size 256 \
                --learning_rate 5e-4 \
                --temperature 5e-2 \
                --eval_every 1 \
                --patience 30 \
                --optimizer RAdam \
                --anneal_every 40 \
                --anneal_factor 0.5 \
                --init_size 5e-2 \
                --num_samples 100000 \
                --sampling adaptive \
                --seed 0 \
                --save_every -1 \
