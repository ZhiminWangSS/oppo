kill -9 $(lsof -t -i :6319)
python testing_agents/test_symbolic_LLMs_capo.py \
--communication \
--prompt_template_path LLM/capo_prompt.csv \
--mode capo_symbolic_gpt4o \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6319 \
--lm_id gpt-4o-ca \
--source openai \
--t 0.7 \
--max_tokens 1024 \
--num_runs 1 \
--num-per-task 2 \