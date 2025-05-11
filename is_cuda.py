import torch

print(torch.cuda.is_available())  # Должно вернуть True, если CUDA доступен
print(torch.cuda.device_count())
