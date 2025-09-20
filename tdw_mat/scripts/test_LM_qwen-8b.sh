lm_id=qwen3-8b
port=10010
pkill -f -9 "port $port"

python3 tdw-gym/challenge_single_capo.py \
--output_dir results \
--lm_id $lm_id \
--experiment_name single_capo-qwen-8b \
--run_id run_1 \
--port $port \
--agents lm_agent \
--prompt_template_path LLM/capo_prompt_single.csv \
--max_tokens 1024 \
--data_prefix dataset/dataset_test/ \
--eval_episodes 0 1 2 10 11 12 13 14 15 21 22 23 \
--screen_size 256 \
--source aliyun

pkill -f -9 "port $port"