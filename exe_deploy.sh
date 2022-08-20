#!/bin/bash
dataset="cifar10"
model="conv"
num_classes=10

python -u main_kde.py --dataset $dataset --model $model --num_classes $num_classes --flag "deploy" | tee "log_kde.txt"

python -u sc.py $dataset $model $num_classes | tee "log_sc.txt"