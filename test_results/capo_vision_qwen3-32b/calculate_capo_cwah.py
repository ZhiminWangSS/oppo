import os
import json
from glob import glob

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 初始化累加器和计数器
sums = {
    "S": 0,
    "L": 0,
    "vision_capo": {
        "episode_0_com_count": 0,
        "episode_1_com_count": 0,
        "episode_0_api": 0,
        "episode_1_api": 0,
        "episode_0_tokens": {
            "communication": {"prompt": 0, "completion": 0, "call_counts": 0},
            "meta-plan": {"prompt": 0, "completion": 0, "call_counts": 0},
            "parsing": {"prompt": 0, "completion": 0, "call_counts": 0},
        },
        "episode_1_tokens": {
            "communication": {"prompt": 0, "completion": 0, "call_counts": 0},
            "meta-plan": {"prompt": 0, "completion": 0, "call_counts": 0},
            "parsing": {"prompt": 0, "completion": 0, "call_counts": 0},
        },
    },
}
count = 0

# 在脚本所在目录下查找所有 .json 文件
json_files = glob(os.path.join(script_dir, "*.json"))

for file_path in json_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 累加 S 和 L（取第一个元素）
        sums["S"] += data["S"][0]
        sums["L"] += data["L"][0]

        vision_capo = data["vision_capo"]
        c_sum = sums["vision_capo"]

        # 累加 vision_capo 下的标量字段
        for key in [
            "episode_0_com_count", "episode_1_com_count",
            "episode_0_api", "episode_1_api"
        ]:
            c_sum[key] += vision_capo[key]

        # 累加嵌套 tokens 字段
        for ep in ["episode_0_tokens", "episode_1_tokens"]:
            for cat in ["communication", "meta-plan","parsing"]:
                for metric in ["prompt", "completion", "call_counts"]:
                    c_sum[ep][cat][metric] += vision_capo[ep][cat][metric]

        count += 1

    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"⚠️ 跳过文件 {file_path}: {e}")
        continue

if count == 0:
    print("❌ 没有找到有效的 JSON 文件")
    exit(1)

# 递归计算平均值的辅助函数
def divide_nested(d, n):
    for k, v in d.items():
        if isinstance(v, dict):
            divide_nested(v, n)
        else:
            d[k] = v / n

averages = {
    "S": [sums["S"] / count],  # 保持为列表格式
    "L": [sums["L"] / count],
    "vision_capo": {}
}

# 深拷贝结构并计算平均值
import copy
avg_vision_capo = copy.deepcopy(sums["vision_capo"])
divide_nested(avg_vision_capo, count)
averages["vision_capo"] = avg_vision_capo

# 保存到脚本所在目录下的 averages.json
output_path = os.path.join(script_dir, "averages.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(averages, f, indent=4, ensure_ascii=False)

print(f"✅ 成功处理 {count} 个文件，均值已保存到 {output_path}")