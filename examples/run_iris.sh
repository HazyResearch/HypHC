#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source set_env.sh
python train.py --dataset iris \
                --epochs 100 \
                --batch_size 512 \
                --learning_rate 5e-4 \
                --temperature 5e-2 \
                --eval_every 1 \
                --patience 20 \
                --optimizer RAdam \
                --anneal_every 30 \
                --anneal_factor 0.5 \
                --init_size 5e-2 \
                --num_samples 50000
