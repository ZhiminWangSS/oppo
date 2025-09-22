lm_id=deepseek-chat
port=10004
pkill -f -9 "port $port"

python3 tdw-gym/challenge.py \
--output_dir results \
--lm_id $lm_id \
--experiment_name LMs-$lm_id \
--run_id run_1 \
--port $port \
--agents lm_agent lm_agent \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--max_tokens 256 \
--data_prefix dataset/dataset_test/ \
--eval_episodes 0 \
--screen_size 256 \
--no_save_img \
--no_gt_mask

pkill -f -9 "port $port"