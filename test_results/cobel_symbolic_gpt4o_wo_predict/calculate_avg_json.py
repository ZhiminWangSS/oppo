import os
import json
from collections import defaultdict

def load_json_files(folder_path='.'):
    """加载指定文件夹下的所有 JSON 文件。"""
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data_list.append(data)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading {filepath}: {e}")
    return data_list

def recursively_calculate_averages(data_list):
    """递归地计算嵌套字典/列表结构中所有数值字段的平均值。"""
    if not data_list:
        return None

    def merge_and_sum(d1, d2, path=""):
        """递归合并两个结构，对数值进行累加，对列表进行拼接（仅当列表元素为数值时求和）。"""
        if isinstance(d1, dict) and isinstance(d2, dict):
            result = {}
            all_keys = set(d1.keys()) | set(d2.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                if key in d1 and key in d2:
                    result[key] = merge_and_sum(d1[key], d2[key], new_path)
                elif key in d1:
                    result[key] = d1[key]
                else: # key in d2
                    result[key] = d2[key]
            return result
        elif isinstance(d1, list) and isinstance(d2, list):
            # 简单处理：如果列表元素都是数字，则求和；否则保留第一个或报错（这里选择保留第一个）
            # 根据你的数据结构，'S' 和 'L' 是单元素列表，我们假设需要平均的是那个元素
            if len(d1) == 1 and len(d2) == 1 and isinstance(d1[0], (int, float)) and isinstance(d2[0], (int, float)):
                return [d1[0] + d2[0]] # 先累加，后面再除以数量
            else:
                # 对于非数值列表或长度不为1的列表，可能需要更复杂的逻辑，这里简单返回第一个
                print(f"Warning: Non-numeric or multi-element list encountered at {path}, using first element.")
                return d1 if d1 else d2
        elif isinstance(d1, (int, float)) and isinstance(d2, (int, float)):
            return d1 + d2
        else:
            # 类型不匹配，返回第一个
            print(f"Warning: Type mismatch at {path}, using first value.")
            return d1

    def divide_by_count(d, count, path=""):
        """递归遍历结构，将累加的数值除以文件数量得到平均值。"""
        if isinstance(d, dict):
            for key in d:
                new_path = f"{path}.{key}" if path else key
                d[key] = divide_by_count(d[key], count, new_path)
            return d
        elif isinstance(d, list):
            if len(d) == 1 and isinstance(d[0], (int, float)):
                return [d[0] / count]
            else:
                return d
        elif isinstance(d, (int, float)):
            return d / count
        else:
            return d

    # 初始化累加器（使用第一个文件的结构）
    accumulator = data_list[0]
    # 从第二个文件开始累加
    for i in range(1, len(data_list)):
        accumulator = merge_and_sum(accumulator, data_list[i])

    # 计算平均值
    average_data = divide_by_count(accumulator, len(data_list))
    return average_data

def main():
    # 读取所有 JSON 文件
    json_data_list = load_json_files('test_results/cobel_symbolic_gpt4o_wo_predict/')
    print(f"Loaded {len(json_data_list)} JSON files.")

    if not json_data_list:
        print("No JSON files found or all files failed to load.")
        return

    # 计算平均值
    averaged_data = recursively_calculate_averages(json_data_list)

    if averaged_data is None:
        print("Failed to calculate averages.")
        return

    # 保存到新文件
    output_filename = 'test_results/cobel_symbolic_gpt4o_wo_predict/averaged_results.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(averaged_data, f, indent=4, ensure_ascii=False)

    print(f"Averaged data saved to '{output_filename}'.")

if __name__ == "__main__":
    main()