import torch
print(torch.__version__)  # 应该输出 2.1.0 或更高

# 测试能否导入 device_mesh
from torch.distributed import device_mesh
print("device_mesh imported successfully")