#model_name_or_path=llava-hf/llava-1.5-7b-hf
#model=llava

model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
model=qwen

# YOUR_DATA_DIR=/home/parekh/VLGuard/
# YOUR_SAVE_DIR=/home/parekh/id_steering/test_code/
# STEER_MODEL_NAME=/home/parekh/xl-vlms/llava_15_last_input_vlguard_train_-1.pt

YOUR_DATA_DIR=/home/parekh/MM-SafetyBench/data/
YOUR_SAVE_DIR=/home/parekh/id_steering/test_code/
#STEER_MODEL_NAME=/home/parekh/id_steering/mmsb_steering_nets/llava_mmsb_steering_net_v3_multi_nobias_K100.pt
STEER_MODEL_NAME=/home/parekh/id_steering/mmsb_steering_nets/qwen_14_last_input_multi_mmsb_train_-1_v2.pt
steer_model_base=$(basename "$STEER_MODEL_NAME" .pt)


data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}


# dataset_name=vlguard
# steering_alpha=2.0

dataset_name=mmsb_test
dataset_size=-1
max_new_tokens=200
steering_alpha=2.2
hook_names=("save_hidden_states_given_token_idx" "shift_hidden_states_learned_steer_no_implicit_model")


for split in multi; do

    for steering_alpha in 2.2; do

        for method in default; do
            shift_vector_path=${STEER_MODEL_NAME}
            save_filename="${model}_${dataset_name}_steer_${method}_${split}_${steering_alpha}_${steer_model_base}.pth"
            #modules_to_hook="language_model.model.layers.30;language_model.model.layers.15" # Used for LLaVA experiments (MMSafety)
            modules_to_hook="model.layers.14;model.layers.14" # Used for Qwen experiments (MMSafety)
            #modules_to_hook="language_model.model.layers.15;language_model.model.layers.15" # Used for LLaVA with VLGuard exps

            python src/save_features.py \
                --model_name_or_path $model_name_or_path \
                --save_dir $save_dir \
                --data_dir $data_dir \
                --split $split \
                --dataset_size $dataset_size \
                --dataset_name $dataset_name \
                --hook_names "${hook_names[@]}" \
                --modules_to_hook "$modules_to_hook" \
                --generation_mode \
                --save_filename $save_filename \
                --local_files_only \
                --exact_match_modules_to_hook \
                --shift_vector_path $shift_vector_path \
                --steering_alpha $steering_alpha \
                --token_idx -1 \
                --max_new_tokens $max_new_tokens \
                --steering_method $method
        done
    done
done
