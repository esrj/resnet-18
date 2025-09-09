import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import log_softmax
from torch import amp
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from data import get_dataloader
from model import ResNet, BasicBlock

def one_hot(labels , num_classes):
    # scatter_(在第幾維操作 , 每列對應的類別位置 , 要填入的值)
    return torch.zeros(labels.size(0), num_classes, device=labels.device).scatter_(1, labels.view(-1,1), 1.)

def soft_cross_entropy(logits, soft_targets):
    # 支援 MixUp/CutMix 的軟標籤
    # 因為 cross entropy 的標籤為定數 ，但 Mix 的標籤為混向量，因此要手動計算，而 cross entropy 不接受混合向量只接受硬標籤
    return torch.mean(torch.sum(-soft_targets * log_softmax(logits, dim=1), dim=1))

# MixUp
def mixup_data(x, y):
    lam = random.uniform(0.5,0.9)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device) # 同一個 batch 隨機打亂順序，如 [0,1,2,3] -> [2,0,1,3]
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam

# Configs
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

num_classes = 100
epochs = 50
base_lr = 1e-3
weight_decay = 1e-4
label_smoothing = 0.1

max_grad_norm = 1.0
min_lr = 3e-5

# Data
train_loader = get_dataloader()  # 你的 train dataloader，建議 shuffle=True

# Model / Optim / Loss
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device).float()
criterion_hard = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

# 把 loss 乘上一個很大的數字（例如 2^16）。
# 這樣 backward 算出來的梯度也一起被放大，不容易 underflow
scaler = amp.GradScaler(device.type)

total_steps = epochs * len(train_loader)
warmup_steps  = int(0.05 * total_steps)
remain_steps  = total_steps - warmup_steps

warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
cosine = CosineAnnealingLR(optimizer, T_max=remain_steps, eta_min=min_lr)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

# Train
for epoch in range(epochs):
    model.train()
    total = 0
    correct = 0

    for x, y in train_loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 依機率使用 MixUp
        use_mix = (torch.rand(1).item() < 0.6)  # 使用 MixUp 的機率 60 %
        if use_mix:
            x_aug, (ya, yb), lam  = mixup_data(x, y)
        else:
            x_aug, ya, yb, lam = x, y, None, 1.0

        with amp.autocast(device.type):
            y_pred = model(x_aug)
            if use_mix:
                # 以軟標籤計算 soft CE
                ya_oh = one_hot(ya, num_classes)
                yb_oh = one_hot(yb, num_classes)
                soft_target = lam * ya_oh + (1 - lam) * yb_oh
                loss = soft_cross_entropy(y_pred, soft_target)
            else:
                loss = criterion_hard(y_pred, ya)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # L2 梯度裁剪
        scaler.step(optimizer) # model 參數更新
        scaler.update() # 動態調整 loss 縮放倍數

        total += y.size(0)
        pred_class = y_pred.argmax(dim=1)  # (B,) 硬預測
        correct += (pred_class == ya).sum().item()  # 和「原始 y」比對

    acc = correct/total
    print(f"Epoch {epoch:02d} | Acc : {acc}")

# Save
model = model.to("cpu")
torch.save(model.state_dict(), "ResNet18.pt")
print("Saved to ResNet18.pt")
