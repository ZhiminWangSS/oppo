kill -9 $(lsof -t -i :6420)
python testing_agents/test_vision_LLMs_cobel.py \
--mode cobel_vision_gpt4o_add3 \
--communication \
--prompt_template_path LLM/cwah_cobel_promptsV2.2_grasp.csv \
--obs_type normal_image \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6420 \
--lm_id gpt-4o-ca \
--source openai \
--t 0.7 \
--max_tokens 512 \
--num_runs 1 \
--num-per-task 2 \
--test_task 10