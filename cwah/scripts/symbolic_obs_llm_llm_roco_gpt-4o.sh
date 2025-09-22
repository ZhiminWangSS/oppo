kill -9 $(lsof -t -i :3000)
python ./testing_agents/test_symbolic_LLMs_roco.py \
--communication \
--prompt_template_path LLM/prompt_roco.csv \
--mode symbolic_roco_gpt-4o \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 3000 \
--lm_id gpt-4o-ca \
--source openai \
--t 0.7 \
--max_tokens 256 \
--num_runs 1 \
--num-per-task 2 \
