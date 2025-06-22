model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava



features_dir=/data/khayatan/Hallucination/POPE/hallucination/features

shift_type=average
save_dir=/data/khayatan/Hallucination/POPE/hallucination/shift_vectors

analysis_name=learnable_steering


for split in adversarial; do

    for i in 14; do

        pos_features_name=save_hidden_states_for_l2s_llava_pope_train_features_pos_answers_14_adversarial_all_train_600.pth
        neg_features_name=save_hidden_states_for_l2s_llava_pope_train_features_neg_answers_14_adversarial_all_train_600.pth


        modules_to_hook="language_model.model.layers.${i}"
        save_filename=${split}_pope_train_-1


        python src/analyse_features.py \
            --model_name_or_path $model_name_or_path \
            --save_dir $save_dir \
            --analysis_name $analysis_name \
            --modules_to_hook $modules_to_hook \
            --save_filename ${save_filename} \
            --local_files_only \
            --shift_type $shift_type \
            --features_path ${features_dir}/${pos_features_name} ${features_dir}/${neg_features_name}
    done
done

# popular random
