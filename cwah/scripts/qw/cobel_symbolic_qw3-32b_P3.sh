kill -9 $(lsof -t -i :6427)
python testing_agents/test_symbolic_LLMs_cobel.py \
--mode cobel_symbolic_qwen3-32b_v3 \
--communication \
--prompt_template_path LLM/cwah_cobel_promptsV2.2_grasp.csv \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 64278 \
--lm_id qwen3-32b \
--source aliyun \
--t 0.7 \
--max_tokens 1024 \
--num_runs 1 \
--num-per-task 2 \
--test_task 30 32 40 49