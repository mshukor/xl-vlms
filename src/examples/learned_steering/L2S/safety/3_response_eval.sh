model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava

YOUR_SAVE_DIR=/home/parekh/id_steering/test_code/
STEER_MODEL_NAME=/home/parekh/id_steering/mmsb_steering_nets/steering_net_v3_multi_nobias_K100.pt
steer_model_base=$(basename "$STEER_MODEL_NAME" .pt)

steering_alpha=2.2
analysis_name="safety_metrics"

#response_filename="qwen_mmsb_test_response_multi_2.2_l2s_temp.pth"
#response_filename="llava_vlguard_response_test_2.0_l2s.pth"
response_filename="llava_mmsb_test_steer_15_multi_2.2_steering_net_v3_multi_nobias_K100.pth"
response_filename="llava_mmsb_test_steer_default_multi_2.2_llava_mmsb_steering_net_v3_multi_nobias_K100.pth"
#response_filename="qwen_mmsb_test_steer__multi_2.2_qwen_14_last_input_multi_mmsb_train_-1_v2.pth"


python src/analyse_features.py \
    --model_name_or_path $model_name_or_path \
    --analysis_name $analysis_name \
    --predictions_path $response_filename \
    --save_filename False