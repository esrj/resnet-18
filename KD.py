import torch
import torch.nn as nn
import torch.optim as optim

from model import ResNet, BasicBlock,Student
from data import get_dataloader

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# 準備資料
train_loader = get_dataloader("cifar100/train")   # 你現成的
test_loader   = get_dataloader("cifar100/test")     # 建議有 val

# 建立 teacher / student
num_classes = 100
teacher = ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)
teacher.load_state_dict(torch.load("ResNet18.pt", map_location="cpu"))
teacher.to(device).eval()          # teacher 只推論，不更新
for p in teacher.parameters():
    p.requires_grad = False  # 關閉 teacher model 的 gradient

student = Student().to(device)

#
T = 4.0
alpha = 0.7
ce_loss = nn.CrossEntropyLoss()   # 強化泛化
kl_div = nn.KLDivLoss(reduction="batchmean")         # KL(student || teacher)

def kd_loss(student_logits, teacher_logits, y):
    # hard label loss
    hard = ce_loss(student_logits, y)
    # soft target loss (用溫度)
    s = torch.log_softmax(student_logits / T, dim=1)
    t = torch.softmax(teacher_logits / T, dim=1)
    soft = kl_div(s, t) * (T * T)
    return (1 - alpha) * hard + alpha * soft


optimizer = optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

epochs = 50
best_val = float("inf")

for epoch in range(epochs):
    student.train()
    total = 0
    correct = 0
    for x, y in train_loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device)

        with torch.no_grad():
            t_logits = teacher(x)

        s_logits = student(x)
        loss = kd_loss(s_logits, t_logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        y_preds = s_logits.argmax(1)
        correct += (y_preds == y).sum().item()
        total += y.size(0)

    scheduler.step()
    acc = correct/total
    print(f"Epoch:{epochs} | Acc {acc}")


student.to("cpu").eval()
torch.save(student.state_dict(), "ResNet18_student_kd.pt")