#!/bin/bash
cd ~/xl-vlms

hook_name=save_hidden_states_for_token_of_interest
token=dog # Can also use other nouns that appear in your dataset
#feature_modules=language_model.model.layers.29
feature_modules=language_model.model.norm,language_model.model.layers.30 # Named modules inside the model for which you want to save the representations
model_name=llava-hf/llava-1.5-7b-hf 
data_dir=/data/mshukor/data/coco/ # Data directory for COCO dataset
split=train # Which data split to save features for. For COCO: train/val/test options
size=82783 # How many samples to consider. karpathy train split for COCO is of size 82783 images
annotation_file=karpathy/dataset_coco.json
save_dir=/home/parekh/

# Other key commands/details:
# (1) We ideally wish to save model representations on train split (to learn the concepts) and test split (to evaluate)
# (2) Use feature modules variable (command --modules_to_hook) to save features for arbitrary layers or other points in the model.
#     You can also save for many named modules at the same time by specifying them one after the other separated by commas
# (3) Other options for hook_name: 
# (4) For representations about different tokens of interest use the --token_of_interest command.
# (5) Ensure you modify dataset path (--data_dir command), and saving directory accordingly (--save_dir command) 


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
--save_filename llava_dog_generation_split_train \
--generation_mode \
--exact_match_modules_to_hook

split=test
size=5000

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
--save_filename llava_dog_generation_split_test \
--generation_mode
