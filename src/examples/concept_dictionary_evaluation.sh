#!/bin/bash
cd ~/github/xl-vlms


model_name=llava-hf/llava-1.5-7b-hf #Specify when using random words (via command --use_random_grounding_words)
analysis_name=concept_dictionary_evaluation_overlap_clipscore_bertscore
feature_module=language_model.model.norm
decomposition=kmeans # Current options: snmf, kmeans, pca, simple
n_concepts=20

features_path=/data/mshukor/logs/xl_vlms/save_hidden_states_for_token_of_interest_llava_yes.pth #/home/parekh/save_hidden_states_for_token_of_interest_llava_train_generation_split_train.pth

python src/analyse_features.py \
--analysis_name $analysis_name \
--features_path $features_path \
--module_to_decompose $feature_module \
--num_concepts $n_concepts \
--decomposition_method $decomposition \
--model_name $model_name \
--save_filename llava_yes


# Additionally you can use --use_random_grounding_words and specify model_name for random words baseline (CLIPScore/BERTscore)

