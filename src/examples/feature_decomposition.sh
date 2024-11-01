#!/bin/bash
cd ~/xl-vlms

analysis_name=decompose_activations_text_grounding_image_grounding
saved_features_path=/home/parekh/save_hidden_states_for_token_of_interest_llava_dog_generation_split_train.pth
#feature_module=language_model.model.norm
feature_module=language_model.model.layers.28
decomposition=snmf # Current options: snmf, kmeans, pca, simple
n_concepts=20

python src/analyse_features.py \
--analysis_name $analysis_name \
--features_path $saved_features_path \
--module_to_decompose $feature_module \
--num_concepts $n_concepts \
--decomposition_method $decomposition \
--save_filename results_dog
