kill -9 $(lsof -t -i :6318)
python testing_agents/test_symbolic_LLMs_cobel.py \
--mode cobel_symbolic_gpt4o_v2 \
--communication \
--prompt_template_path LLM/cwah_cobel_promptsV2.2_grasp.csv \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6318 \
--lm_id gpt-4o-ca \
--source openai \
--t 0.7 \
--max_tokens 1024 \
--num_runs 1 \
--num-per-task 2 \
