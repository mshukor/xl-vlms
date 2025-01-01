#!/bin/bash

YOUR_XL_VLM_DIR=YOUR_XL_VLM_DIR
YOUR_XL_VLM_DIR=/home/khayatan/xl_vlms_cvpr/xl-vlms
YOUR_VQA_DIR=YOUR_VQA_DIR
YOUR_VQA_DIR=/data/mshukor/data/coco/

model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava

# model_name_or_path=HuggingFaceM4/idefics2-8b
# model=idefics2

# model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
# model=qwen2vlinstruct

# model_name_or_path=allenai/Molmo-7B-D-0924
# model=molmo


dataset_name=vqav2
data_dir=YOUR_VQA_DIR
dataset_size=5000
save_feats_dir=${YOUR_XL_VLM_DIR}/results/features
save_dir=${YOUR_XL_VLM_DIR}/results
max_new_tokens=5

steering_method=shift_of_means
# steering_method=mean_of_shifts

steering_alpha=1
analysis_name=steering_vector
start_prompt_token_idx_steering=0
token_of_interest=""
layers=(31)


save_dataset_size=2000
category_of_interest=no
token_of_interests=("yes" "no")

# save_dataset_size=5000
# category_of_interest=number
# token_of_interests=("1" "3")

# save_dataset_size=5000
# category_of_interest=other
# token_of_interests=("white" "black")


for i in "${!layers[@]}"; do
    ## compute steering vector

    layer=${layers[$i]}
    modules_to_hook=language_model.model.layers.${layer}
    ## save features across layers
    hook_names=save_hidden_states_for_token_of_interest

    python src/analyse_features.py \
    --model_name_or_path $model_name_or_path \
    --analysis_name $analysis_name \
    --steering_method $steering_method \
    --base_features_key ${hook_names}_${model}_${layer}_${token_of_interests[0]}.pth \
    --features_path ${save_feats_dir}/${hook_names}_${model}_${layer}_${token_of_interests[0]}.pth ${save_feats_dir}/${hook_names}_${model}_${layer}_${token_of_interests[1]}.pth \
    --module_to_decompose $modules_to_hook \
    --save_dir $save_dir \
    --save_filename ${model}_${layer}_${token_of_interests[0]}_to_${token_of_interests[1]} \
    --local_files_only
done