#!/bin/bash

python3 train.py --dataset breast_cancer \
                 --log_path ../logs/breast_cancer_train_log \
                 --model b2 \
                 --num_steps 400 \
                 --feature_extract 0 \
                 --num_classes 3 \
                 --batch_size 32 \
                 --gpu_id 2 \
                 --augment 0 \
                 --lr 0.0001 \
                 --num_workers 4 \
                 --patience 20 \
                 --eval_freq 10


python3 train.py --dataset breast_cancer \
                 --log_path ../logs/breast_cancer_train_log \
                 --model b2 \
                 --num_steps 400 \
                 --feature_extract 0 \
                 --num_classes 3 \
                 --batch_size 32 \
                 --gpu_id 2 \
                 --augment 0 \
                 --lr 0.0001 \
                 --num_workers 4 \
                 --patience 20 \
                 --eval_freq 10
