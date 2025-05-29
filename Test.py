import torch
print("CUDA доступно:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Устройство:", torch.cuda.get_device_name(0))
    print("Версия CUDA:", torch.version.cuda)
print("Версия PyTorch:", torch.__version__)