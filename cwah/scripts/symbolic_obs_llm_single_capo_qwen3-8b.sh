kill -9 $(lsof -t -i :6310)
python3 ./testing_agents/test_symbolic_LLM_single_capo.py \
--mode single_LLM_capo_qwen3-8b \
--prompt_template_path LLM/capo_prompt_single.csv \
--agent_num 1 \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6310 \
--lm_id qwen3-8b \
--source aliyun \
--t 0.7 \
--max_tokens 512 \
--num_runs 1 \
--num-per-task 2 \
