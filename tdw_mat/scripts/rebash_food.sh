#!/bin/bash

# ===== 配置区 =====
SCRIPT="./scripts/cobel_food_gpt4o_twoup.sh"   # 👈 修改成你的脚本路径
SLEEP_SECONDS=3             # 异常退出后等待几秒再重启
# ==================

echo "[$(date)] 监控启动：$SCRIPT"

while true; do
    echo "[$(date)] 正在执行脚本..."
    
    # 执行你的脚本
    "$SCRIPT"
    exit_code=$?
    
    if [ $exit_code -eq 1 ]; then
        echo "[$(date)] 脚本正常退出（退出码 0），监控结束。"
        break
    else
        echo "[$(date)] 脚本异常退出（退出码 $exit_code），${SLEEP_SECONDS}秒后重启..."
        sleep $SLEEP_SECONDS
    fi
done