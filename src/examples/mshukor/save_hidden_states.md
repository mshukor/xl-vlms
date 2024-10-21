## Save hidden states

Save all hidden states in teacher forcing mode:
```
python src/save_features.py \
--model_name llava-hf/llava-1.5-7b-hf \
--modules_to_hook language_model.model.layers.6.mlp.up_proj language_model.model.layers.1.mlp.up_proj \
--save_dir /data/mshukor/logs/xl_vlms \
--save_filename llava_hidden_states

```
Save hidden states given a token index:

```
python src/save_features.py \
--model_name llava-hf/llava-1.5-7b-hf \
--modules_to_hook language_model.model.layers.6.mlp.up_proj language_model.model.layers.1.mlp.up_proj \
--save_dir /data/mshukor/logs/xl_vlms \
--save_filename llava_hidden_states \
--hook_name save_hidden_states_given_token_idx \
--token_idx 36
```

Save hiddens states of tokens between start and end tokens:

```
python src/save_features.py \
--model_name llava-hf/llava-1.5-7b-hf \
--modules_to_hook language_model.model.layers.6.mlp.up_proj language_model.model.layers.1.mlp.up_proj \
--save_dir /data/mshukor/logs/xl_vlms \
--save_filename llava_hidden_states \
--hook_name save_hidden_states_given_token_start_end_idx \
--token_start_end_idx 36 40
```

Load hidden states for analysis:

```
python src/analyse_features.py --features_path /data/mshukor/logs/xl_vlms/save_hidden_states_given_token_start_end_idx_llava_hidden_states.pth
```


Save hidden states for COCO dataset:
```
python src/save_features.py \
--model_name llava-hf/llava-1.5-7b-hf \
--modules_to_hook language_model.model.layers.6.mlp.up_proj language_model.model.layers.1.mlp.up_proj \
--save_dir /data/mshukor/logs/xl_vlms \
--save_filename llava_hidden_states \
--hook_name save_hidden_states_given_token_start_end_idx \
--token_start_end_idx 36 40 \
--data_dir /data/mshukor/data/coco/ \
--annotation_file karpathy/dataset_coco.json \
--split val \
--dataset_size 100 \
--dataset_name coco
```

### Save TOI

To save hidden states for the token dog
```
python src/save_features.py \
--model_name llava-hf/llava-1.5-7b-hf \
--save_dir /data/mshukor/logs/xl_vlms \
--save_filename llava_dog_generation \
--data_dir /data/mshukor/data/coco/ \
--annotation_file karpathy/dataset_coco.json \
--split val \
--dataset_size 200 \
--dataset_name coco \
--hook_name save_hidden_states_for_token_of_interest \
--select_token_of_interest_samples \
--modules_to_hook language_model.model.norm \
--token_of_interest dog \
--generation_mode
```
