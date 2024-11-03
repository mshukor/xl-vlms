#!/bin/bash

# Path to your xl-vlms repository
cd ~/xl-vlms

# Need to specify analysis type. This will evaluate overlap, clipscore and bertscore metrics
analysis_name=concept_dictionary_evaluation_overlap_clipscore_bertscore

# For evaluation, specify path where you have saved features on test data
test_features_path=/home/parekh/save_hidden_states_for_token_of_interest_llava_train_generation_split_test.pth
#features_path=/data/mshukor/logs/xl_vlms/save_hidden_states_for_token_of_interest_llava_yes.pth

# Specify the feature module for which concept dictionary was extracted
feature_module=language_model.model.norm
#feature_module=language_model.model.layers.30

# Specify the details for saved concept dictionary
decomposition_path=results/decompose_activations_text_grounding_image_grounding_results_train.pth  # Filepath of saved results

# Additionally you can use --use_random_grounding_words and specify model_name for random words baseline (CLIPScore/BERTscore)
model_name=llava-hf/llava-1.5-7b-hf

# --local_files_only to load hf models from local cache

python src/analyse_features.py \
--analysis_name $analysis_name \
--features_path $test_features_path \
--module_to_decompose $feature_module \
--model_name $model_name \
--save_filename llava_yes \
--local_files_only \
--concepts_decomposition_path $decomposition_path \
#--use_random_grounding_words


