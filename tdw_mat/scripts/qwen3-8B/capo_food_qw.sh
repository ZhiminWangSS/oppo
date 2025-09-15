lm_id=qwen3-8b
port=11108
pkill -f -9 "port $port"

python3 tdw-gym/challenge_capo.py \
--output_dir results/qw/capo_qwen3-8b_food \
--lm_id $lm_id \
--experiment_name LMs-$lm_id \
--run_id run_1 \
--port $port \
--agents lm_agent_capo lm_agent_capo \
--communication \
--debug \
--prompt_template_path LLM/capo_prompt.csv \
--max_tokens 1024 \
--data_prefix dataset/dataset_test/ \
--eval_episodes 1 2 3 4 5 6 7 8 9 10 11 \
--screen_size 256 \
--source aliyun
pkill -f -9 "port $port"