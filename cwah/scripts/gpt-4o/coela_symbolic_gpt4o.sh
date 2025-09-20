kill -9 $(lsof -t -i :6414)
python testing_agents/test_symbolic_LLMs.py \
--mode colea_symbolic_debug \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6414 \
--lm_id gpt-4o-ca \
--source openai \
--t 0.7 \
--max_tokens 256 \
--num_runs 1 \
--num-per-task 2 \
--debug