import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloader
from model import ResNet, BasicBlock
import torch_pruning as tp


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dataloader = get_dataloader(path = 'cifar100/train')

# 載 model
model = torch.jit.load("ResNet18_structured_pruned.ts", map_location=device)

print(" ================= testing =================")

with torch.no_grad():
    model.eval()
    total = 0
    correct = 0
    for x,y in dataloader:
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        y_class = torch.argmax(y_pred,dim = 1)

        total += len(y)
        correct += (y_class == y).sum().item()
    acc = correct/total

    print(f"Accuracy : {acc}")

