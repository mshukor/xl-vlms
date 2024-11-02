#!/bin/bash

# Path to your xl-vlms repository
cd ~/xl-vlms

# --hook_name controls strategy of feature extraction
hook_name=save_hidden_states_for_token_of_interest  # Only save features for a token of interest in the output
#hook_name=save_hidden_states_given_token_start_end_idx   # Save all representations from start to end indices
#hook_name=save_hidden_states_given_token_idx   # Save representations at specific index 

token=train 
#token=dog # Can also use other nouns that appear in your dataset

# Directory and filename to store extracted features
results_filename=llava_train_generation_split_train
save_dir=/home/parekh/

model_name=llava-hf/llava-1.5-7b-hf

# Named modules inside the model for which you want to save the representations
feature_modules=language_model.model.norm,language_model.model.layers.30
# Other examples of named modules
#feature_modules=language_model.model.layers.28.input_layernorm
#feature_modules=language_model.model.layers.29
 
# Dataset specifications. Ensure you modify dataset path (--data_dir command) accordingly
data_dir=/data/mshukor/data/coco/ # Data directory for COCO dataset
split=train # Which data split to save features for. For COCO: train/val/test options
size=82783 # How many samples of dataset to consider. karpathy train split for COCO is of size 82783 images. Can't be more than dataset size
annotation_file=karpathy/dataset_coco.json


python src/save_features.py \
--model_name $model_name \
--dataset_name coco \
--dataset_size $size \
--data_dir $data_dir \
--annotation_file $annotation_file \
--split $split \
--hook_name $hook_name \
--modules_to_hook $feature_modules \
--select_token_of_interest_samples \
--token_of_interest $token \
--save_dir $save_dir \
--save_filename $results_filename \
--generation_mode \
--exact_match_modules_to_hook


# We save model representations on both train split (to learn the concepts) and test split (to evaluate)
split=test
size=5000
results_filename=llava_train_generation_split_test

python src/save_features.py \
--model_name $model_name \
--dataset_name coco \
--dataset_size $size \
--data_dir $data_dir \
--annotation_file $annotation_file \
--split $split \
--hook_name $hook_name \
--modules_to_hook $feature_modules \
--select_token_of_interest_samples \
--token_of_interest $token \
--save_dir /home/parekh/ \
--save_filename $results_filename \
--generation_mode \
--exact_match_modules_to_hook
