model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava



features_dir=/data/khayatan/Hallucination/POPE/hallucination/features

shift_type=average
save_dir=/data/khayatan/Hallucination/POPE/hallucination/shift_vectors

analysis_name=learnable_steering


for split in adversarial popular random; do

    for i in 14; do

        pos_features_name=save_hidden_states_for_l2s_llava_pope_test_features_pos_answers_${i}_${split}_-1.pth
        neg_features_name=save_hidden_states_for_l2s_llava_pope_test_features_neg_answers_${i}_${split}_-1.pth


        modules_to_hook="language_model.model.layers.${i};language_model.model.layers.${i}"
        save_filename=${split}_pope_test_-1


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
/data/khayatan/Hallucination/POPE/hallucination/shift_vectors/llava_14_average_random_pope_test_-1.pth
"""