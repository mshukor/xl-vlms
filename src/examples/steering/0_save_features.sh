#!/bin/bash


model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava

dataset_name=vqav2
data_dir=/data/mshukor/data/coco/
answer_type_to_answer=/data/mshukor/data/coco/type_to_answer_dict.json
dataset_size=5000
predictions_path=/home/khayatan/xl_vlms_cvpr/xl-vlms/results/features/vqav2_accuracy_${model}_vqav2_val_baseline_datasize_${dataset_size}_model_prediction.json
max_new_tokens=5

steering_method=shift_of_means
# steering_method=mean_of_shifts
steering_hook_name=shift_hidden_states_add
# steering_hook_name=shift_hidden_states_add_projected
# steering_hook_name=shift_hidden_states_add_only_generated
# steering_hook_name=shift_hidden_states_add_last_prompt_token

# start_prompt_token_idx_steering=577
# steering_hook_name=shift_hidden_states_add_start_idx
steering_alpha=1
analysis_name=steering_vector
start_prompt_token_idx_steering=0
token_of_interest=""
# layers=(31 29 27 25 23 21 19 15 11 7 3 1)
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
    layer=${layers[$i]}
    modules_to_hook=language_model.model.layers.${layer}
    ## save features across layers
    hook_names=save_hidden_states_for_token_of_interest
    split=train2014
    annotation_file=v2_mscoco_${split}_annotations.json
    questions_file=v2_OpenEnded_mscoco_${split}_questions.json
    for i in "${!token_of_interests[@]}"; do
        token_of_interest=${token_of_interests[$i]}
        python src/save_features.py \
        --model_name_or_path $model_name_or_path \
        --data_dir $data_dir \
        --annotation_file $annotation_file \
        --questions_file $questions_file \
        --split $split \
        --dataset_size $save_dataset_size \
        --dataset_name $dataset_name \
        --hook_names $hook_names \
        --save_only_generated_tokens \
        --select_token_of_interest_samples \
        --modules_to_hook $modules_to_hook \
        --token_of_interest_key response \
        --token_of_interest $token_of_interest \
        --generation_mode \
        --save_filename ${model}_${layer}_${token_of_interest} \
        --local_files_only \
        --exact_match_modules_to_hook
    done
done