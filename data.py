import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_mean_std(path = 'cifar100/train'):
    dataset = ImageFolder(root=path,transform=transforms.ToTensor())
    img_stack = []
    for img,label in dataset:
        img_stack.append(img)  # [3, 32, 32]


    #  [3, 32, 32, N]
    img_stack = torch.stack(img_stack, dim = 3)
    # [3 , 32*32*N ]
    mean = img_stack.reshape(3,-1).mean(dim = 1)
    std = img_stack.reshape(3,-1).std(dim = 1)

    return tuple(mean),tuple(std)

def get_dataloader(path = 'cifar100/train'):
    mean,std = get_mean_std(path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    dataset = ImageFolder(root=path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    return dataloader



