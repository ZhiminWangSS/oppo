kill -9 $(lsof -t -i :6515)
python testing_agents/test_symbolic_LLMs_cobel_multi.py \
--mode cobel_symbolic_gpt4o_ma_num3 \
--communication \
--prompt_template_path LLM/cwah_cobel_promptsV2.2_grasp.csv \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6515 \
--lm_id gpt-4o-ca \
--source openai \
--t 0.7 \
--max_tokens 1024 \
--num_runs 1 \
--num-per-task 2 \
--agent_num 3 \
--test_task 0 5 10
