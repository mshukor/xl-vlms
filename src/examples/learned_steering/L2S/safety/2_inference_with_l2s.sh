model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava

YOUR_DATA_DIR=/home/parekh/MM-SafetyBench/data/
YOUR_SAVE_DIR=/home/parekh/id_steering/test_code/
STEER_MODEL_NAME=/home/parekh/id_steering/mmsb_steering_nets/steering_net_v3_multi_nobias_K100.pt
steer_model_base=$(basename "$STEER_MODEL_NAME" .pt)


data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}


dataset_name=mmsb_test
dataset_size=-1
max_new_tokens=200
steering_alpha=2.2
hook_names=("save_hidden_states_given_token_idx" "shift_hidden_states_learned_steer")


for split in multi; do

    for steering_alpha in 2.2; do

        for i in 15; do
            shift_vector_path=${STEER_MODEL_NAME}
            save_filename="${model}_${dataset_name}_steer_${i}_${split}_${steering_alpha}_${steer_model_base}"
            modules_to_hook="language_model.model.layers.30;language_model.model.layers.${i}"

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
                --max_new_tokens $max_new_tokens
        done
    done
done
