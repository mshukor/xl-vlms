model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava

YOUR_DATA_DIR=/data/khayatan/datasets/POPE/test
YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination


data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}


save_dir=/data/khayatan/Hallucination/POPE/hallucination
dataset_name=pope_test
dataset_size=-1

max_new_tokens=100
hook_names=("captioning_metrics")



for split in adversarial popular random; do
    save_filename="${model}_${dataset_name}_eval_no_steer_${split}_${dataset_size}"
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
        --generation_mode \
        --save_filename ${save_filename} \
        --local_files_only \
        --descriptive_answer \
        --end_special_tokens "</s>"
done


