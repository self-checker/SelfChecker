#!/bin/bash
python -u main_kde.py  | tee "log_kde.txt" 

for((i=0;i<10;i++)); 
do
	python layer_selection_agree.py $((i)) &
done

wait


for((i=0;i<10;i++)); 
do
	python layer_selection_condition.py $((i)) &
done

wait

for((i=0;i<10;i++)); 
do
	python layer_selection_condition_neg.py $((i)) &
done

# for((i=0;i<10;i++)); 
# do

# 	for((j=0;j<10;j++)); 
# do
# 	python layer_selection_agree.py $((j+i*10)) &
# done

# wait
# done

# for((i=0;i<10;i++)); 
# do

# 	for((j=0;j<10;j++)); 
# do
# 	python layer_selection_condition.py $((j+i*10)) &
# done

# wait
# done

# for((i=0;i<10;i++)); 
# do

# 	for((j=0;j<10;j++)); 
# do
# 	python layer_selection_condition_neg.py $((j+i*10)) &
# done

# wait
# done

wait

python -u sc.py  | tee "log_sc.txt"