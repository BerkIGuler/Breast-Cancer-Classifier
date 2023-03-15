#!/bin/bash

python3 train.py --dataset breast_cancer \
                 --log_path ../logs/breast_cancer_log \
                 --model b2 \
                 --num_steps 4000 \
                 --feature_extract 0 \
                 --num_classes 3 \
                 --batch_size 32 \
                 --gpu_id 2 \
                 --augment 0 \
                 --lr 0.0001 \
                 --num_workers 4 \
                 --patience 20 \
                 --eval_freq 250 \
                 --original_dataset_path /auto/data2/bguler/DDAN/breast_cancer/train \
                 --ddpm_dataset_path /auto/data2/bguler/DDAN/breast_cancer_ddpm1.0 \
                 --original_batch_size 16 \
                 --ddpm_batch_size 16 \
