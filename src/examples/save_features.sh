#!/bin/bash
cd /home/parekh/xl-vlms

hook_name=save_hidden_states_for_token_of_interest
token=train # Can also use other noun that appear in your dataset
feature_modules=language_model.model.norm 
model_name=llava-hf/llava-1.5-7b-hf
data_dir=/data/mshukor/data/coco/
split=train
size=82783 # How many samples to consider

python src/save_features.py \
--model_name $model_name \
--dataset_name coco \
--dataset_size $size \
--data_dir $data_dir \
--annotation_file karpathy/dataset_coco.json \
--split $split \
--hook_name $hook_name \
--modules_to_hook $feature_modules \
--select_token_of_interest_samples \
--token_of_interest $token \
--save_dir /home/parekh/ \
--save_filename llava_train_generation_split_train \
--generation_mode

#split=test
#size=5000

#python src/save_features.py \
#--model_name $model_name \
#--dataset_name coco \
#--dataset_size $size \
#--data_dir $data_dir \
#--annotation_file karpathy/dataset_coco.json \
#--split $split \
#--hook_name $hook_name \
#--modules_to_hook $feature_modules \
#--select_token_of_interest_samples \
#--token_of_interest $token \
#--save_dir /home/parekh/ \
#--save_filename llava_train_generation_split_test \
#--generation_mode

