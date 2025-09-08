import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloader
from model import ResNet, BasicBlock

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_loader = get_dataloader()  # 確認這個是 train dataloader，且 shuffle=True
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100).to(device).float()

epochs = 25
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # 0.1 smoothing 後可能變成 [0.033,0.033,0.9,0.033]。
optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4) # weight_decay 把所有的係數往 0 拉一點

for epoch in range(epochs):
    model.train()
    running_correct = 0
    running_total = 0
    running_loss = 0.0

    for x, y in train_loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # 記錄訓練 acc/loss
        preds = logits.argmax(dim=1)
        running_total += y.size(0)
        running_correct += (preds == y).sum().item()
        running_loss += loss.item() * y.size(0)

    acc = running_correct / running_total
    avg_loss = running_loss / running_total
    print(f"Epoch {epoch:02d} | loss: {avg_loss:.4f} | acc: {acc:.4f}")

# 儲存
model = model.to("cpu")
torch.save(model.state_dict(), 'ResNet18.pt')