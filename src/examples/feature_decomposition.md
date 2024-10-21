## Save hidden states for token of interest


```
token_of_interest=dog
modules_to_hook=language_model.model.norm
# to load the model from local you can set --local_files_only 
python src/save_features.py \
--model_name llava-hf/llava-1.5-7b-hf \
--save_dir /data/mshukor/logs/xl_vlms \
--data_dir /data/mshukor/data/coco/ \
--annotation_file karpathy/dataset_coco.json \
--split val \
--dataset_size 200 \
--dataset_name coco \
--save_filename llava_dog \
--hook_name save_hidden_states_for_token_of_interest \
--select_token_of_interest_samples \
--modules_to_hook $modules_to_hook \
--token_of_interest $token_of_interest \
--generation_mode \
--local_files_only 
```

## Feature decomposition

Decompose features of a module, averaged over all stored positions with Semi-NMF (5 concepts)
```
decomposition_method=snmf
num_concepts=10
modules_to_hook=language_model.model.norm
python src/analyse_features.py \
--analysis_name decompose_activations \
--features_path /home/parekh/save_hidden_states_given_token_start_end_idx_llava_hidden_states.pth \
--module_to_decompose $modules_to_hook \
--num_concepts $num_concepts \
--decomposition_method $decomposition_method \
--local_files_only 
```

Decompose features of a module, for a specific stored position with KMeans (10 concepts)
```
python src/analyse_features.py \
--analysis_name decompose_activations \
--features_path /home/parekh/save_hidden_states_given_token_start_end_idx_llava_hidden_states.pth \
--module_to_decompose language_model.model.layers.6.mlp.up_proj \
--decomposition_extract_pos 0 \
--num_concepts 10 \
--decomposition_method kmeans
```

Decompose samples from COCO with dog TOI and perform multimodal grounding:

```
python src/analyse_features.py \
--analysis_name decompose_activations_text_grounding_image_grounding \
--features_path /data/mshukor/logs/xl_vlms/save_hidden_states_for_token_of_interest_llava_dog_generation.pth \
--module_to_decompose language_model.model.norm --num_concepts 9 --decomposition_method snmf
```
