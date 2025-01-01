#!/bin/bash

YOUR_XL_VLM_DIR=YOUR_XL_VLM_DIR
YOUR_XL_VLM_DIR=/home/khayatan/xl_vlms_cvpr/xl-vlms
YOUR_VQA_DIR=YOUR_VQA_DIR
YOUR_VQA_DIR=/data/mshukor/data/coco/
YOUR_ANSWER_TYPE_TO_ANSWER_FILE=YOUR_ANSWER_TYPE_TO_ANSWER_FILE
YOUR_ANSWER_TYPE_TO_ANSWER_FILE=/data/mshukor/data/coco/type_to_answer_dict.json


save_dir=${YOUR_XL_VLM_DIR}/results


model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava

# model_name_or_path=HuggingFaceM4/idefics2-8b
# model=idefics2

# model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
# model=qwen2vlinstruct

# model_name_or_path=allenai/Molmo-7B-D-0924
# model=molmo


dataset_name=vqav2
data_dir=${YOUR_VQA_DIR}
dataset_size=5000
answer_type_to_answer=${YOUR_ANSWER_TYPE_TO_ANSWER_FILE}
max_new_tokens=5

python src/save_features.py \
--model_name_or_path $model_name_or_path \
--save_dir $save_dir \
--save_filename ${model}_vqav2_val_baseline_datasize_${dataset_size} \
--data_dir ${YOUR_VQA_DIR} \
--annotation_file v2_mscoco_val2014_annotations.json \
--questions_file v2_OpenEnded_mscoco_val2014_questions.json \
--split val2014 \
--dataset_size $dataset_size \
--dataset_name $dataset_name \
--hook_name vqav2_accuracy \
--generation_mode \
--local_files_only \
--token_of_interest no \
--category_of_interest no \
--max_new_tokens $max_new_tokens \
--answer_type_to_answer $answer_type_to_answer