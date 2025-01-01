#!/bin/bash


YOUR_COCO_DIR=YOUR_COCO_DIR
YOUR_COCO_DIR=/data/mshukor/data/coco/
YOUR_COCO_ANNOTATION_FILE=YOUR_COCO_ANNOTATION_FILE
YOUR_COCO_ANNOTATION_FILE=karpathy/dataset_coco.json


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

steering_method=shift_of_means
steering_alpha=1
analysis_name=steering_vector
start_prompt_token_idx_steering=0
token_of_interest=""
# layers=(31 29 27 25 23 19 15 11 7 3 1)
layers=(19)


## save general features across layers
hook_names=save_hidden_states_before_special_tokens
split=train
annotation_file=${YOUR_COCO_ANNOTATION_FILE}
save_dataset_size=2000
for i in "${!layers[@]}"; do
    layer=${layers[$i]}
    modules_to_hook=language_model.model.layers.${layer}
    python src/save_features.py \
    --model_name_or_path $model_name_or_path \
    --data_dir $data_dir \
    --annotation_file $annotation_file \
    --split $split \
    --dataset_size $save_dataset_size \
    --dataset_name $dataset_name \
    --hook_names $hook_names \
    --save_only_generated_tokens \
    --modules_to_hook $modules_to_hook \
    --generation_mode \
    --save_filename ${model}_coco_${layer}_class_all \
    --local_files_only \
    --exact_match_modules_to_hook \
    --allow_different_variations_of_token_of_interest \
    --end_special_tokens "</s>"
done

save_dataset_sizes=(3000)
token_of_interest_classes=("colors")

# save_dataset_sizes=(2000)
# token_of_interest_classes=("places")

# save_dataset_sizes=(25000)
# token_of_interest_classes=("sentiments")



## save style features across layers
for j in "${!token_of_interest_classes[@]}"; do
    token_of_interest_class=${token_of_interest_classes[$j]}
    save_dataset_size=${save_dataset_sizes[$j]}
    token_of_interest=$token_of_interest_class
    for i in "${!layers[@]}"; do
        layer=${layers[$i]}
        modules_to_hook=language_model.model.layers.${layer}
        hook_names=save_hidden_states_for_token_of_interest_class
        save_filename=${model}_coco_${layer}_onlytoi_class_${token_of_interest_class}

        ## save features for each class across layers
        split=train
        annotation_file=${YOUR_COCO_ANNOTATION_FILE}
        python src/save_features.py \
        --model_name_or_path $model_name_or_path \
        --data_dir $data_dir \
        --annotation_file $annotation_file \
        --split $split \
        --dataset_size $save_dataset_size \
        --dataset_name $dataset_name \
        --hook_names $hook_names \
        --save_only_generated_tokens \
        --select_token_of_interest_samples \
        --modules_to_hook $modules_to_hook \
        --token_of_interest_key response \
        --token_of_interest $token_of_interest \
        --token_of_interest_class $token_of_interest_class \
        --generation_mode \
        --save_filename $save_filename \
        --local_files_only \
        --exact_match_modules_to_hook \
        --end_special_tokens "</s>"
    done
done
