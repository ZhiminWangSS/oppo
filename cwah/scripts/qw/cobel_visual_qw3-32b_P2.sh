kill -9 $(lsof -t -i :6436)
python testing_agents/test_vision_LLMs_cobel.py \
--mode cobel_vision_qwen3-32b_v2 \
--communication \
--prompt_template_path LLM/cwah_cobel_promptsV2.2_grasp.csv \
--obs_type normal_image \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6436 \
--lm_id qwen3-32b \
--source aliyun \
--t 0.7 \
--max_tokens 1024 \
--num_runs 1 \
--num-per-task 2 \
--test_task 16 20 26 