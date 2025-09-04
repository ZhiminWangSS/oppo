import pickle

# 替换为你的 .pik 文件路径
file_path = '/home/aiseon/zmwang/codes/cobel/oppo/test_results/LLMs_comm_gpt-4/logs_agent_0_read_book_0.pik'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 查看内容
print("数据类型:", type(data))
print("数据内容:\n", data)