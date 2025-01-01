#!/bin/bash

YOUR_XL_VLM_DIR=YOUR_XL_VLM_DIR
YOUR_XL_VLM_DIR=/home/khayatan/xl_vlms_cvpr/xl-vlms
YOUR_COCO_DIR=YOUR_COCO_DIR
YOUR_COCO_DIR=/data/mshukor/data/coco/
YOUR_COCO_ANNOTATION_FILE=YOUR_COCO_ANNOTATION_FILE
YOUR_COCO_ANNOTATION_FILE=karpathy/dataset_coco.j

model_name_or_path=llava-hf/llava-1.5-7b-hf

dataset_name=coco
data_dir=${YOUR_COCO_DIR}
dataset_size=50
save_steer_dir=${YOUR_XL_VLM_DIR}/results/steering
save_dir=${YOUR_XL_VLM_DIR}/results
max_new_tokens=25

steering_method=shift_of_means
steering_hook_name=save_hidden_states_shift_hidden_states_add
steering_alpha=1
analysis_name=steering_vector
start_prompt_token_idx_steering=0
token_of_interest=""
# layers=(31 29 27 25 23 19 15 11 7 3 1)
layers=(19)


annotation_file=${YOUR_COCO_ANNOTATION_FILE}


token_of_interest_classes=("colors")

# token_of_interest_classes=("places")

# token_of_interest_classes=("sentiments")

for j in "${!token_of_interest_classes[@]}"; do
    token_of_interest_class=${token_of_interest_classes[$j]}
    token_of_interest=$token_of_interest_class
    for i in "${!layers[@]}"; do
        layer=${layers[$i]}
        modules_to_hook=language_model.model.layers.${layer}

  
        ## steering
        split=val
        shift_vector_path=${save_steer_dir}/${steering_method}_llava_coco_${layer}_class_all_to_${token_of_interest_class}_onlytoi.pth
        save_filename=${steering_method}_llava_coco_${layer}_class_all_to_${token_of_interest_class}_onlytoi_${shift_vector_key}_${steering_hook_name}
        shift_vector_key=steering_vector
        hook_names=($steering_hook_name "captioning_metrics")
        python src/save_features.py \
        --model_name_or_path $model_name_or_path \
        --save_dir $save_dir \
        --data_dir $data_dir \
        --annotation_file $annotation_file \
        --split $split \
        --dataset_size $dataset_size \
        --dataset_name $dataset_name \
        --hook_names "${hook_names[@]}" \
        --modules_to_hook $modules_to_hook \
        --token_of_interest $token_of_interest \
        --generation_mode \
        --save_filename $save_filename \
        --local_files_only \
        --exact_match_modules_to_hook \
        --shift_vector_path $shift_vector_path \
        --shift_vector_key $shift_vector_key \
        --steering_alpha $steering_alpha \
        --max_new_tokens $max_new_tokens \
        --start_prompt_token_idx_steering $start_prompt_token_idx_steering

    done
done