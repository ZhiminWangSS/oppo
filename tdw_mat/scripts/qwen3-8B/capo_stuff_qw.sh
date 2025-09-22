lm_id=qwen3-8b
port=11198
pkill -f -9 "port $port"

python3 tdw-gym/challenge_capo.py \
--output_dir results/qw/capo_qwen3-8b_stuff \
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
--eval_episodes 12 13 14 15 16 17 18 19 20 21 22 23 \
--screen_size 256 \
--source aliyun \
--no_save_img
pkill -f -9 "port $port"