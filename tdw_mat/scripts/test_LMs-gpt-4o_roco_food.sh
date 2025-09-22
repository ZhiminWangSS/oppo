lm_id=gpt-4o-ca
port=1004
pkill -f -9 "port $port"

python3 tdw-gym/challenge_roco.py \
--output_dir results_roco_food \
--lm_id $lm_id \
--experiment_name LMs-$lm_id \
--run_id run_1 \
--port $port \
--agents lm_agent_roco lm_agent_roco \
--communication \
--prompt_template_path LLM/roco_prompt.csv \
--max_tokens 1024 \
--data_prefix dataset/dataset_test/ \
--eval_episodes 0 1 2 3 4 5 6 7 8 9 10 11 \
--screen_size 256 \
--no_save_img

pkill -f -9 "port $port"