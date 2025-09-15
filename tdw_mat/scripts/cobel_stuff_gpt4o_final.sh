lm_id=gpt-4o-ca
port=10098
pkill -f -9 "port $port"

#!/bin/bash

# 获取脚本所在目录 cd 到tdw_mat目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 计算项目根目录（tdw_mat目录）
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 设置路径
WORKSPACE_DIR="$PROJECT_ROOT"  # 就是tdw_mat目录本身
BELIEF_SYMBOLIC_DIR="$(dirname "$PROJECT_ROOT")"

# 设置Python路径
export PYTHONPATH="${WORKSPACE_DIR}:${BELIEF_SYMBOLIC_DIR}:${PYTHONPATH}"

cd "$WORKSPACE_DIR"

echo "当前工作目录: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"


python3 tdw-gym/challenge_cobel.py \
--output_dir results/gpt4o_stuff_final \
--lm_id $lm_id \
--experiment_name LMs-$lm_id \
--run_id run_0914 \
--port $port \
--agents lm_agent_cobel lm_agent_cobel \
--communication \
--debug \
--prompt_template_path LLM/cobel_promptsV2.1_no_conf_two_update_hardencode.csv \
--max_tokens 1024 \
--data_prefix dataset/dataset_test/ \
--eval_episodes 12 13 14 21 22 23 \
--screen_size 256
pkill -f -9 "port $port"