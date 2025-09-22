kill -9 $(lsof -t -i :10004)
python testing_agents/test_vision_LLMs_roco.py \
--mode roco_vision_gpt_4o\
--communication \
--prompt_template_path LLM/prompt_roco.csv \
--obs_type normal_image \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 10004 \
--lm_id qwen3-32b \
--source aliyun \
--t 0.7 \
--max_tokens 1024 \
--num_runs 1 \
--num-per-task 2 \
