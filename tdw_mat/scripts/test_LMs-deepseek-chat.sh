lm_id=deepseek-chat
port=10008
pkill -f -9 "port $port"

python3 tdw-gym/challenge_oppo.py \
--output_dir results_capo_ds-v3/retest \
--experiment_name LMs-$lm_id \
--run_id run_1 \
--data_prefix dataset/dataset_test/ \
--port $port \
--agents lm_agent_capo lm_agent_capo \
--eval_episodes 3 8 9 14 17 19 22 23 \
--communication \
--debug \
--source deepseek \
--lm_id $lm_id \
--prompt_template_path LLM/capo_prompt.csv \
--max_tokens 2048 \
--screen_size 256 \
--no_save_img 

pkill -f -9 "port $port"