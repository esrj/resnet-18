import torch
from data import get_dataloader

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

dataloader = get_dataloader(path = 'cifar100/test')

# # 載 pruning model
# model = torch.jit.load("ResNet18_structured_pruned.ts", map_location=device)

# # 載 model
from model import ResNet, BasicBlock
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100).to(device).float()
model.load_state_dict(torch.load("ResNet18.pt"))

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
