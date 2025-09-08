import torch
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp
# custom module
from model import ResNet, BasicBlock
from data import get_dataloader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 載入模型
model = ResNet(BasicBlock, [2,2,2,2], num_classes=100).to(device)
model.load_state_dict(torch.load("ResNet18.pt", map_location="cpu"))

# 建立一個隨機輸入讓 Torch-Pruning 前向傳播。這樣它就知道哪一個 layer 的輸出會接到哪一個 layer 的輸入
example_inputs = torch.randn(1, 3, 32, 32).to(device)  # CIFAR-100 input size
DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# 設定剪枝比例 (例如 Conv2d 通道剪掉 30%)
prune_ratio = 0.3
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        # CNN kernel 的權重為 (out_channel,in_channel,H,W)
        # 實際代表的意義為 ( kernel 數量 , kernel channel , kernel H , kernel W )
        weight = m.weight.detach().abs().mean(dim=(1,2,3)) # input channel、高度、寬度 三個維度取平均。
        num_prune = int(prune_ratio * weight.shape[0]) # 要剪幾個
        if num_prune < 1:
            continue

        # torch.argsort(weight) 回傳權重強度由小到大排序的 index。
        prune_index = torch.argsort(weight)[:num_prune].tolist()

        # tp.prune_conv_out_channel：告訴 Torch-Pruning 要剪掉的是 kernel 數量
        # prune_index 你要剪的 kernel 編號清單
        group = DG.get_pruning_group(m, tp.prune_conv_out_channels, prune_index)
        if DG.check_pruning_group(group):  # 檢查是否安全可剪
            group.prune()

print('==================================================')
print('fine-ture')
train_loader = get_dataloader()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
epochs = 10
for epoch in range(epochs):
    model.train()
    correct = 0
    total = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        y_class = logits.argmax(dim=1)
        correct += (y_class == y).sum().item()
        total += y.size(0)

    print(f"Epoch {epoch}:  acc={correct/total:.4f}")

model.eval().to("cpu")
scripted = torch.jit.script(model)
torch.jit.save(scripted, "ResNet18_structured_pruned.ts")