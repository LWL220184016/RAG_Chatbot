import torch

def check_gpu():
    print(torch.cuda.is_available())
    print("Num GPUs Available: ", torch.cuda.device_count())

    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    cpus = ["CPU"]  # PyTorch does not have a direct method to list CPUsconda 
    print(gpus, cpus)

if __name__ == "__main__":
    check_gpu()