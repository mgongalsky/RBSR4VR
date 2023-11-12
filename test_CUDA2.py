import torch

# Проверяем, доступен ли CUDA (GPU)
cuda_available = torch.cuda.is_available()
print(f"CUDA (GPU) доступен: {cuda_available}")

# Выводим текущее устройство по умолчанию
default_device = torch.device("cuda" if cuda_available else "cpu")
print(f"Текущее устройство: {default_device}")
