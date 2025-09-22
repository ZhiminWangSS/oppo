kill -9 $(lsof -t -i :6455)
python ./testing_agents/test_symbolic_LLMs_roco.py \
--communication \
--prompt_template_path LLM/prompt_roco.csv \
--mode roco_symbolic_qwen3-32b \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6455 \
--lm_id qwen3-32b \
--source aliyun \
--t 0.7 \
--max_tokens 256 \
--num_runs 1 \
--num-per-task 2 \
--test_task 0 5 10