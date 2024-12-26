#!/bin/bash

# Path to your xl-vlms repository
cd ~/xl-vlms

finetuning=color # CHANGE TO PLACE OR SENTIMENT OR COLOR ACCORDING TO THE EXPERIMENT
TOI=dog

# To analyze how the clusters are shifted using shift vectors, and to save the statistics about the recovery of concepts
analysis_name=analyse_clusters
save_filename="${finetuning}_${TOI}_analysis"


# Model specifications. change the cache directory accordingly
model_name_or_path=llava-hf/llava-1.5-7b-hf
cache_dir=/data/khayatan/llava/

# Extracted hidden states from llava model
origin_model_feature_path=src/assets/llava_converted_hidden_states/save_hidden_states_original_${finetuning}.pth
dest_model_feature_path=src/assets/llava_converted_hidden_states/save_hidden_states_${finetuning}.pth
module_to_decompose=language_model.model.norm


# Hidden states decomposition and grounding
num_concepts=20
num_grounded_text_tokens=15
pre_num_top_tokens=200 # number of text groundings before filtering them (so that we can have at least "num_grounded_text_tokens" meaningful text groundings at the end)
num_most_activating_samples=50
decomposition_method=kmeans

# Dataset specifications. Ensure you modify dataset path (--data_dir command) accordingly
dataset_name=coco
data_dir=/data/mshukor/data/coco/ # Data directory for COCO dataset
split=train # Which data split to save features for. For COCO: train/val/test options
annotation_file=karpathy/dataset_coco.json
# path to the ids of the samples
path_to_samples_ids=src/assets/concepts_ids/${finetuning}.pkl


python src/analyse_features.py \
--model_name_or_path $model_name_or_path \
--dataset_name $dataset_name \
--data_dir $data_dir \
--annotation_file $annotation_file \
--split $split \
--cache_dir $cache_dir \
--select_samples_from_ids \
--path_to_samples_ids $path_to_samples_ids \
--analysis_name $analysis_name \
--save_filename $save_filename \
--origin_model_feature_path $origin_model_feature_path \
--dest_model_feature_path $dest_model_feature_path \
--module_to_decompose $module_to_decompose \
--num_concepts $num_concepts \
--num_grounded_text_tokens $num_grounded_text_tokens \
--pre_num_top_tokens $pre_num_top_tokens \
--num_most_activating_samples $num_most_activating_samples \
--decomposition_method $decomposition_method \
--visualize_concepts \
--compute_recovery_metrics \
--compute_stat_shift_vectors
