#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source set_env.sh
python train.py --dataset glass \
                --epochs 50 \
                --batch_size 512 \
                --learning_rate 5e-4 \
                --temperature 1e-1 \
                --eval_every 1 \
                --patience 40 \
                --optimizer RAdam \
                --anneal_every 10 \
                --anneal_factor 0.75 \
                --init_size 0.05 \
                --num_samples 100000 \
                --fast_decoding true \
                --seed 0
