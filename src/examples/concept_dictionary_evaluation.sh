#!/bin/bash
cd /home/parekh/xl-vlms

eval_name=overlap_clipscore_bertscore
concept_dict_path=/home/parekh/xl-vlms/results/snmf_results_train.pth
saved_test_features_path=/home/parekh/save_hidden_states_for_token_of_interest_llava_train_generation_split_test.pth
feature_module=language_model.model.norm
model_name=llava-hf/llava-1.5-7b-hf #Specify when using random words (via command --use_random_words)

python src/evaluate_concepts.py \
--evaluation_name $eval_name \
--decomposition_path  $concept_dict_path \
--module_to_decompose $feature_module \
--features_path $saved_test_features_path


# Additionally use --use_random_words and specify model_name for random words baseline (CLIPScore/BERTscore)

#python src/evaluate_concepts.py \
#--evaluation_name $eval_name \
#--decomposition_path  $concept_dict_path \
#--module_to_decompose language_model.model.norm \
#--features_path $saved_test_features_path \
#--use_random_words \
#--model_name $model_name
