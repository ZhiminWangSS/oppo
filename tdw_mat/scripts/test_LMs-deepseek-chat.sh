lm_id=deepseek-chat
port=10008
pkill -f -9 "port $port"

python3 tdw-gym/challenge_coela.py \
--output_dir results_coela_chat_coela_0726 \
--lm_id $lm_id \
--experiment_name LMs-$lm_id \
--run_id run_1 \
--port $port \
--agents lm_agent lm_agent \
--communication \
--debug \
--prompt_template_path LLM/prompt_com.csv \
--max_tokens 512 \
--data_prefix dataset/dataset_test/ \
--eval_episodes 7 8 10 14 19 20 \
--screen_size 256
pkill -f -9 "port $port"