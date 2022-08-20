#!/bin/bash
dataset="cifar10"
model="conv"
num_classes=10
num_layers=9

python -u main_kde.py --dataset $dataset --model $model --num_classes $num_classes | tee "log_kde_train.txt"

for((i=0;i<$num_classes;i++));
do
	python layer_selection_agree.py $dataset $model $num_classes $num_layers $((i)) &
done

wait


for((i=0;i<$num_classes;i++));
do
	python layer_selection_condition.py $dataset $model $num_classes $num_layers $((i)) &
done

wait

for((i=0;i<$num_classes;i++));
do
	python layer_selection_condition_neg.py $dataset $model $num_classes $num_layers $((i)) &
done

wait
