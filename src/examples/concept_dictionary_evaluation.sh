#!/bin/bash
cd ~/xl-vlms

# Specify the path to saved features and the name of layer
#feature_module=language_model.model.layers.28
feature_module=language_model.model.norm

#features_path=/data/mshukor/logs/xl_vlms/save_hidden_states_for_token_of_interest_llava_yes.pth #/home/parekh/save_hidden_states_for_token_of_interest_llava_train_generation_split_train.pth
features_path=/home/parekh/save_hidden_states_for_token_of_interest_llava_train_generation_split_test.pth
decomposition_path=results/simple_results_train.pth

model_name=llava-hf/llava-1.5-7b-hf
analysis_name=concept_dictionary_evaluation_overlap_clipscore_bertscore
decomposition=simple # Current options: snmf, kmeans, pca, simple
n_concepts=20


python src/analyse_features.py \
--analysis_name $analysis_name \
--features_path $features_path \
--module_to_decompose $feature_module \
--num_concepts $n_concepts \
--decomposition_method $decomposition \
--model_name $model_name \
--save_filename llava_yes \
--local_files_only \
--concepts_decomposition_path $decomposition_path

# --local_files_only to load hf models from local
# Additionally you can use --use_random_grounding_words and specify model_name for random words baseline (CLIPScore/BERTscore)
