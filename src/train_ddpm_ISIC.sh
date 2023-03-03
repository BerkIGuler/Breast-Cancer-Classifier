#!/bin/bash

python3 train.py --dataset breast_cancer \
                 --log_path ./log \
                 --model b2 \
                 --num_epochs 500 \
                 --feature_extract 0 \
                 --num_classes 3 \
                 --batch_size 32 \
                 --gpu_id 2 \
                 --augment 1 \
                 --lr 0.001 \
                 --num_workers 4 \
                 --patience 20 \
                 --eval_freq 100