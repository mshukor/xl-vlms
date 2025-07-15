model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava
YOUR_DATA_DIR=/data/khayatan/datasets/POPE/train
YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination

data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}


dataset_name=pope_train
dataset_size=-1

features_dir=/data/khayatan/Hallucination/POPE/hallucination/features

shift_type=average
save_dir=/data/khayatan/Hallucination/POPE/hallucination/shift_vectors

analysis_name=learnable_steering


for split in all; do

    for i in 14; do

        pos_features_name=save_hidden_states_for_l2s_llava_pope_train_features_pos_answers_14_${split}_all_train_${dataset_size}.pth
        neg_features_name=save_hidden_states_for_l2s_llava_pope_train_features_neg_answers_14_${split}_all_train_${dataset_size}.pth

        modules_to_hook="language_model.model.layers.${i};language_model.model.layers.${i}"

        save_filename=${split}_pope_train_-1


        python src/analyse_features.py \
            --model_name_or_path $model_name_or_path \
            --save_dir $save_dir \
            --analysis_name $analysis_name \
            --modules_to_hook $modules_to_hook \
            --save_filename ${save_filename} \
            --local_files_only \
            --shift_type $shift_type \
            --features_path ${features_dir}/${pos_features_name} ${features_dir}/${neg_features_name} \
            --seed 0
    done
done




"""
Saving individual shift vectors in : 
/data/khayatan/Hallucination/POPE/hallucination/shift_vectors/llava_14_average_all_pope_train_-1.pth
Saving mean shift vectors in : 
/data/khayatan/Hallucination/POPE/hallucination/shift_vectors/llava_14_average_all_pope_train_-1_mean.pth

"""
