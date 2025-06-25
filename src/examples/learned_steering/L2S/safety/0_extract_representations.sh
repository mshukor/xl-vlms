model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava

YOUR_DATA_DIR=/home/parekh/MM-SafetyBench/data/
YOUR_SAVE_DIR=/home/parekh/id_steering/test_code/

data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}

dataset_name=mmsb
dataset_size=-1

max_new_tokens=1



hook_names=("save_hidden_states_for_l2s")
modules_to_hook=""

# split multi will extract steering vectors according to multi-behavior prompt completions for MM-Safety depending upon input scenario
# split single will extract steering vectors using a single prompt completion for all examples corresponding to safety

for split in multi; do

    for i in 15; do

        modules_to_hook="model.layers.${i}"
        modules_to_hook="language_model.model.layers.${i}"
        save_pos_filename="${model}_${dataset_name}_features_pos_answers_${i}_${split}_all_${dataset_size}"
        save_neg_filename="${model}_${dataset_name}_features_neg_answers_${i}_${split}_all_${dataset_size}"
        save_cxt_filename="${model}_${dataset_name}_features_context_${i}_${split}_all_${dataset_size}"

        # First command computes positive answer representations
        # Second command computes negative answer representations
        # Third command computes input context representations

        python src/save_features.py \
            --model_name_or_path $model_name_or_path \
            --data_dir $data_dir \
            --dataset_name $dataset_name \
            --split $split \
            --annotation_file annotations.json \
            --dataset_size $dataset_size \
            --save_dir $save_dir \
            --max_new_tokens $max_new_tokens \
            --hook_names $hook_names \
            --modules_to_hook $modules_to_hook \
            --save_filename ${save_pos_filename} \
            --local_files_only \
            --force_answer \
            --forced_answer_true \
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>"
        
        python src/save_features.py \
            --model_name_or_path $model_name_or_path \
            --data_dir $data_dir \
            --dataset_name $dataset_name \
            --dataset_size $dataset_size \
            --split $split \
            --save_dir $save_dir \
            --max_new_tokens $max_new_tokens \
            --hook_names $hook_names \
            --modules_to_hook $modules_to_hook \
            --save_filename ${save_neg_filename} \
            --local_files_only \
            --force_answer \
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>"
        
        python src/save_features.py \
            --model_name_or_path $model_name_or_path \
            --data_dir $data_dir \
            --dataset_name $dataset_name \
            --dataset_size $dataset_size \
            --split $split \
            --save_dir $save_dir \
            --max_new_tokens $max_new_tokens \
            --hook_names $hook_names \
            --modules_to_hook $modules_to_hook \
            --save_filename ${save_cxt_filename} \
            --local_files_only \
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>"
    done
done

