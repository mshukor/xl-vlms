## Feature decomposition

Decompose features of a module, averaged over all stored positions with Semi-NMF (5 concepts)
```
python src/analyse_features.py \
--analysis_name decompose_activations \
--features_path /home/parekh/save_hidden_states_given_token_start_end_idx_llava_hidden_states.pth \
--module_to_decompose language_model.model.layers.6.mlp.up_proj \
--num_concepts 5 \
--decomposition_method snmf
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
