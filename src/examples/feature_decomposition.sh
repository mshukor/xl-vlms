#!/bin/bash

# Path to your xl-vlms repository
cd ~/xl-vlms

#model_name=llava-hf/llava-1.5-7b-hf
model_name=allenai/Molmo-7B-D-0924

# Need to specify analysis type. This will decompose features, and extract both text, visual grounding
analysis_name=decompose_activations_text_grounding_image_grounding

# Specify path where you have saved features on training data
saved_features_path=/home/parekh/features/save_hidden_states_for_token_of_interest_molmo_train_generation_split_train.pth

# Where to store details about extracted concepts. Default directory is results/
results_filename=results_train

# Specify the specific feature module you want to decompose
# Molmo
#feature_module=model.transformer.blocks.27
feature_module=model.transformer.ln_f

# LLaVA-v1.5
#feature_module=language_model.model.norm
#feature_module=language_model.model.layers.30

decomposition=snmf # Current options: snmf, kmeans, pca, simple
n_concepts=20 # Size of dictionary learnt i.e. number of concepts

python src/analyse_features.py \
--model_name $model_name \
--analysis_name $analysis_name \
--features_path $saved_features_path \
--module_to_decompose $feature_module \
--num_concepts $n_concepts \
--decomposition_method $decomposition \
--save_filename $results_filename
