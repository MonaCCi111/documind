import torch

print(f'Доступна CUDA: {torch.cuda.is_available()}')
print(f'Кол-во GPU: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Название GPU: {torch.cuda.get_device_name(0)}')