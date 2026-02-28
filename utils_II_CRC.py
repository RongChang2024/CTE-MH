import os
import pickle
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import roc_auc_score

# ==============================
# 训练/验证主循环（二分类专用）
# ==============================

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, scheduler):
    model.train()
    total_loss = 0
    all_labels = []
    all_probs = []

    for step, (image_ids, images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(images)
            logits = logits.view(-1)  # 或 logits = logits.squeeze(1)
            loss = criterion(logits, labels)    # 这里要缩进，写进with块里

        loss.backward()
        optimizer.step()
        scheduler.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        total_loss += loss.item()

    auc = roc_auc_score(all_labels, all_probs)
    avg_loss = total_loss / (step + 1)
    print(f'[train epoch {epoch}] loss: {avg_loss:.4f}, auc: {auc:.4f}')
    return avg_loss, auc

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, epoch, split):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []

    for step, (image_ids, images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.float().to(device)

        with autocast():
            logits = model(images)
            logits = logits.view(-1)  # 或 logits = logits.squeeze(1)
            loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        total_loss += loss.item()

    auc = roc_auc_score(all_labels, all_probs)
    avg_loss = total_loss / (step + 1)
    print(f'[{split} epoch {epoch}] loss: {avg_loss:.4f}, auc: {auc:.4f}')
    return avg_loss, auc

# ==============================
# 其他实用函数（可选使用）
# ==============================

def plot_data_loader_image(data_loader):
    import matplotlib.pyplot as plt
    plot_num = min(data_loader.batch_size, 4)
    for _, images, labels in data_loader:
        for i in range(plot_num):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * 0.5) + 0.5  # 反归一化
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.title(str(label))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('float32'))
        plt.show()
        break

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)

def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def create_lr_scheduler(optimizer,
                       num_step: int,
                       epochs: int,
                       warmup=True,
                       warmup_epochs=1,
                       warmup_factor=1e-3,
                       end_factor=1e-6):
    import math
    assert num_step > 0 and epochs > 0
    if not warmup:
        warmup_epochs = 0
    def f(x):
        if warmup and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
