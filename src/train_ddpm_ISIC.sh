#!/bin/bash

python3.6 train.py --dataset ISIC_300_ddpm --model b2 --num_epochs 92 --feature_extract 0 --num_classes 8 --batch_size 32 --gpu_id 0 --augment 1 --lr 0.001 --num_workers 4 --patience 100 --eval_freq 100
python3.6 train.py --dataset ISIC_300_ddpm --model b2 --num_epochs 92 --feature_extract 0 --num_classes 8 --batch_size 32 --gpu_id 0 --augment 1 --lr 0.001 --num_workers 4 --patience 100 --eval_freq 100
python3.6 train.py --dataset ISIC_300_ddpm --model b2 --num_epochs 92 --feature_extract 0 --num_classes 8 --batch_size 32 --gpu_id 0 --augment 1 --lr 0.001 --num_workers 4 --patience 100 --eval_freq 100





