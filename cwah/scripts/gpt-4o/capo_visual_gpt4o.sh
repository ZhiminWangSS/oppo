kill -9 $(lsof -t -i :6418)
python testing_agents/test_vision_LLMs.py \
--mode capo_vision_gpt4o \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--obs_type normal_image \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6418 \
--lm_id gpt-4o-ca \
--source openai \
--t 0.7 \
--max_tokens 1024 \
--num_runs 1 \
--num-per-task 2 \
--debug