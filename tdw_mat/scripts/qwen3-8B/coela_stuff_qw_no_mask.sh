lm_id=qwen3-8b
port=11078
pkill -f -9 "port $port"

python3 tdw-gym/challenge.py \
--output_dir results/qw/coela_qwen3-8b_food_no_mask \
--lm_id $lm_id \
--experiment_name LMs-$lm_id \
--run_id run_1 \
--port $port \
--agents lm_agent lm_agent \
--communication \
--debug \
--no_gt_mask \
--prompt_template_path LLM/prompt_com.csv \
--max_tokens 1024 \
--data_prefix dataset/dataset_test/ \
--eval_episodes 12 13 14 15 16 17 18 19 20 21 22 23 \
--screen_size 256 \
--source aliyun
pkill -f -9 "port $port"1529