model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava

YOUR_DATA_DIR=/data/khayatan/datasets/POPE/test
YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination
YOUR_SHIFTS_PATH="/data/khayatan/Hallucination/POPE/hallucination/shift_vectors/llava_${i}_average_${subset}_test_all.pth"


data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}


dataset_name=pope_test
dataset_size=-1
max_new_tokens=100
steering_alpha=1
hook_names=("shift_hidden_states_add" "captioning_metrics")
shift_vector_key=steering_vector


for subset in adversarial popular random; do

    for steering_alpha in 1; do

        for i in 14; do
            shift_vector_path=${YOUR_SHIFTS_PATH}
            save_filename="${model}_${dataset_name}_steer_${i}_yes_no_${subset}_${steering_alpha}_p2s"
            modules_to_hook="language_model.model.layers.${i}"


            python src/save_features.py \
                --model_name_or_path $model_name_or_path \
                --save_dir $save_dir \
                --data_dir $data_dir \
                --split $subset \
                --dataset_size $dataset_size \
                --dataset_name $dataset_name \
                --hook_names "${hook_names[@]}" \
                --modules_to_hook $modules_to_hook \
                --generation_mode \
                --save_filename $save_filename \
                --local_files_only \
                --exact_match_modules_to_hook \
                --shift_vector_path $shift_vector_path \
                --shift_vector_key $shift_vector_key \
                --steering_alpha $steering_alpha \
                --individual_shift \
                --max_new_tokens $max_new_tokens
        done
    done
done
