#!/bin/bash

YOUR_XL_VLM_DIR=YOUR_XL_VLM_DIR
YOUR_COCO_DIR=YOUR_COCO_DIR
YOUR_COCO_ANNOTATION_FILE=YOUR_COCO_ANNOTATION_FILE

model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava

# model_name_or_path=HuggingFaceM4/idefics2-8b
# model=idefics2

# model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
# model=qwen2vlinstruct

# model_name_or_path=allenai/Molmo-7B-D-0924
# model=molmo

dataset_name=coco
data_dir=${YOUR_COCO_DIR}
dataset_size=3000
save_feats_dir=${YOUR_XL_VLM_DIR}/results/features
save_dir=${YOUR_XL_VLM_DIR}/results
max_new_tokens=25

steering_method=shift_of_means
steering_alpha=1
analysis_name=steering_vector
start_prompt_token_idx_steering=0
token_of_interest=""
# layers=(31 29 27 25 23 19 15 11 7 3 1)
layers=(19)



token_of_interest_classes=("colors")

# token_of_interest_classes=("places")

# token_of_interest_classes=("sentiments")

for j in "${!token_of_interest_classes[@]}"; do
    token_of_interest_class=${token_of_interest_classes[$j]}
    for i in "${!layers[@]}"; do
        layer=${layers[$i]}
        modules_to_hook=language_model.model.layers.${layer}
        hook_names=save_hidden_states_for_token_of_interest_class
     
        ## compute steering vector
        target_features_path=${save_feats_dir}/${hook_names}_${model}_coco_${layer}_onlytoi_class_${token_of_interest_class}.pth
        save_filename=${model}_coco_${layer}_class_all_to_${token_of_interest_class}_onlytoi
        base_features_key=save_hidden_states_before_special_tokens_${model}_coco_${layer}_class_all.pth
        python src/analyse_features.py \
        --model_name_or_path $model_name_or_path \
        --save_dir $save_dir \
        --analysis_name $analysis_name \
        --steering_method $steering_method \
        --base_features_key $base_features_key \
        --features_path ${save_feats_dir}/${base_features_key} $target_features_path \
        --module_to_decompose $modules_to_hook \
        --save_filename $save_filename \
        --local_files_only

    done
done

