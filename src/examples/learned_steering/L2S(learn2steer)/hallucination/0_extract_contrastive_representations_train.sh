#!/bin/bash                                                                                                        
#SBATCH --job-name=hallucination_pope_train_llava_extract_all_layers   # Job name                                                         
#SBATCH --nodes=1                                                                                                  
#SBATCH --gpus-per-node=1 
#SBATCH --time=1-10                # Maximum runtime (D-HH:MM:SS)                                                  
#SBATCH -p electronic,hard                # Partition or queue name                                                
#SBATCH --cpus-per-task=40                                                                                          
#SBATCH --output=%x-%j.out                                                                                          
#SBATCH                              # This is an empty line to separate Slurm directives from the job commands    
                                                                                                                   
echo "Start Job $SLURM_ARRAY_TASK_ID on $HOSTNAME"  # Display job start information                                
                                                                                                                   
sleep 10  # Sleep for 10 seconds                                                                                   
                                                                                                                   
source activate /data/khayatan/envs/XAI                                                                      
                                                                                                                   
sleep 10  # Sleep for 10 seconds   

# #SBATCH --mem-per-cpu=30                                                                                         




model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava


YOUR_DATA_DIR=/data/khayatan/datasets/POPE/test
YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination
YOUR_SHIFTS_PATH="/data/khayatan/Hallucination/POPE/hallucination/shift_vectors/llava_${i}_average_${subset}_test_all.pth"


data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}


save_dir=/data/khayatan/Hallucination/POPE/hallucination
dataset_name=pope_train
dataset_size=-1
dataset_size=600

max_new_tokens=100


hook_names=("save_hidden_states")
modules_to_hook=""


for split in adversarial popular random; do

    for i in 14; do

        modules_to_hook="model.layers.${i}"
        modules_to_hook="language_model.model.layers.${i}"
        save_filename="${model}_${dataset_name}_features_right_answers_${i}_${split}_all_train_${dataset_size}"


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
            --generation_mode \
            --save_filename ${save_filename} \
            --local_files_only \
            --force_answer \
            --forced_answer_true \
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>"
    done
done



for split in adversarial popular random; do

    for i in 14; do

        modules_to_hook="model.layers.${i}"
        modules_to_hook="language_model.model.layers.${i}"
        save_filename="${model}_${dataset_name}_features_wrong_answers_${i}_${split}_all_train_${dataset_size}"


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
            --generation_mode \
            --save_filename ${save_filename} \
            --local_files_only \
            --force_answer \
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>"
    done
done