# By Deqing Song

import os
import numpy as np
import pandas as pd
import random
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics import ConfusionMatrix

import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append("Process_Data")
from Data2Models import create_tensor_dataset_tcnn

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
path_name = os.path.dirname(os.path.abspath(__file__))+'/TrainedModels/'

num_features = 4
num_classes = 4
time_window = 28

batch_size = 64
lr = 0.001
max_epochs = 100  # 最大训练 Epoch，EarlyStopping 可提前停止

network_type = '2L3DTCNN'
normalization = False  # 如果需要归一化可手动设置

# ==================================================
# 1) 定义 EarlyStopping 类
# ==================================================
class EarlyStopping:
    """基于验证集 Loss 的提前停止."""
    def __init__(self, patience=10, min_delta=1e-4):
        """
        参数:
            patience (int): 验证集 Loss 若在连续多少个 epoch 内没有提升，则停止。
            min_delta (float): 判定“提升”的最小阈值。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# ==================================================
# 2) 定义 3D 时间卷积网络 (TCNN) 模型
# ==================================================
class Time3DCNNSequence(nn.Module):
    def __init__(self, network_type, num_classes=5, num_features=4, time_window=28):
        super(Time3DCNNSequence, self).__init__()

        if network_type == '1L3DTCNN': # RT2DCNN
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=32,
                                   kernel_size=(28, 3, 3), stride=1, padding=0)
            self.global_max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(32, num_classes)

        elif network_type == '2L3DTCNN': # RT3DCNN
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=16,
                                   kernel_size=(5, 3, 3), stride=1, padding=0)
            self.conv2 = nn.Conv3d(in_channels=16, out_channels=32,
                                   kernel_size=(5, 1, 1), stride=1, padding=0)
            self.global_max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(32, num_classes)

        self.network_type = network_type

    def forward(self, input):
        # input 形状: (batch, time_window, num_features, ? ...)
        # 需要先扩展出 channel 维度
        x = input.unsqueeze(1)

        if self.network_type == '1L3DTCNN':
            x = nn.functional.relu(self.conv1(x))
            x = self.global_max_pool(x)
            x = self.flatten(x)
            x = self.fc(x)

        elif self.network_type == '2L3DTCNN':
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.relu(self.conv2(x))
            x = self.global_max_pool(x)
            x = self.flatten(x)
            x = self.fc(x)
        return x

# ==================================================
# 3) 在给定数据集上做推理的工具函数
# ==================================================
def get_output(data_ds, model, device):
    labels_pred = []
    model.eval()
    with torch.no_grad():
        for i in range(len(data_ds.data_target)):
            x, y = data_ds.__getitem__(i)
            x = x.unsqueeze(0).to(device)  # 扩展 batch 维度
            logits = model(x)
            logits = logits.squeeze()
            labels_pred.append(logits.cpu().numpy())

    labels_pred = np.array(labels_pred)           # (N, num_classes)
    labels_pred = labels_pred.argmax(axis=1)
    labels_true = np.array(data_ds.data_target[:])
    labels_true = labels_true.astype('int64')
    return torch.tensor(labels_pred), torch.tensor(labels_true)

# ==================================================
# 4) 主函数
# ==================================================
if __name__ == '__main__':

    # 固定随机种子
    torch.manual_seed(2020)
    np.random.seed(2020)
    random.seed(2020)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------
    # 4.1) 载入整个数据集
    # -----------------------------
    full_dataset = create_tensor_dataset_tcnn(
        main_path + 'DATA/3_labeled_window_dataset_post123.csv',
        num_classes=num_classes
    )

    total_size = len(full_dataset)
    print(f"Total samples in dataset: {total_size}")

    # -----------------------------
    # 4.2) 拆分成训练/验证/测试
    #     先从 full_dataset 中拿 20% 做测试集
    # -----------------------------
    test_size = int(0.2 * total_size)
    train_val_size = total_size - test_size

    train_val_data, test_data = random_split(
        full_dataset,
        [train_val_size, test_size],
        generator=torch.Generator().manual_seed(2020)
    )

    # 再把 train_val_data 中的 20% 用作验证集
    val_size = int(0.2 * train_val_size)
    train_size = train_val_size - val_size

    train_data, val_data = random_split(
        train_val_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(2021)
    )

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}, Test samples: {len(test_data)}")

    # -----------------------------
    # 4.3) 构建 DataLoader
    # -----------------------------
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    # 查看一个 batch 的形状
    train_features, train_labels_ = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels_.size()}")

    # -----------------------------
    # 4.4) 构建模型 & 优化器
    # -----------------------------
    model = Time3DCNNSequence(network_type, num_classes=num_classes).double().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # -----------------------------
    # 4.5) Early Stopping 初始化
    # -----------------------------
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())

    # 用于绘制 Train/Val Loss 曲线
    train_loss_history = []
    val_loss_history   = []

    # -----------------------------
    # 4.6) 训练循环
    # -----------------------------
    for epoch in range(max_epochs):
        # ---- (a) 训练阶段 ----
        model.train()
        running_train_loss = []
        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss.append(loss.item())

        epoch_train_loss = np.mean(running_train_loss)
        train_loss_history.append(epoch_train_loss)

        # ---- (b) 验证阶段 ----
        model.eval()
        running_val_loss = []
        with torch.no_grad():
            for X_batch, y_batch in val_dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                running_val_loss.append(loss.item())

        epoch_val_loss = np.mean(running_val_loss)
        val_loss_history.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{max_epochs}] "
              f"- Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}")

        # 如果验证集 Loss 更优，则更新 best_weights
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = copy.deepcopy(model.state_dict())

        # ---- (c) Early Stopping 检查 ----
        if early_stopping(epoch_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}. "
                  f"Best Val Loss: {best_val_loss:.4f}")
            break

    # -----------------------------
    # 4.7) 恢复验证集最优权重
    # -----------------------------
    model.load_state_dict(best_weights)
    print("Loaded best model weights (based on val loss).")

    # -----------------------------
    # 4.8) 绘制 Train/Val Loss 曲线
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Curve')
    plt.legend()
    plt.grid()
    plt.show()

    # -----------------------------
    # 4.9) 在测试集上评估性能 (Confusion Matrix)
    # -----------------------------
    model.eval()
    confusionMatrixMetric = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    # 在 test_dataloader 上推理
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred_logits = model(X_batch)
            preds = y_pred_logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y_batch.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    cm = confusionMatrixMetric(all_labels, all_preds)
    print("Confusion Matrix on Test Set:\n", cm)

    plt.figure()
    if num_classes == 5:
        label_classes = ["NC", "ST", "DT", "P", "G"]
    elif num_classes == 4:
        label_classes = ["NC", "ST", "P", "G"]
    sns.heatmap(cm.numpy(), annot=True, fmt='d', cmap='Blues',
                xticklabels=label_classes, yticklabels=label_classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    plt.show()

    # -----------------------------
    # 4.10) 是否保存模型
    # -----------------------------
    named_tuple = time.localtime() 
    if input('do you want to save the data in trained models? (y/n):') == 'y':
        tag = input('Please put a tag if needed: ')
        output_path = (path_name + network_type +
                       time.strftime("_%m_%d_%Y_%H-%M-%S", named_tuple) +
                       tag + '.pth')

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "network_type": network_type,
            "n_epochs": epoch + 1,  # 训练到第几个 epoch
            "batch_size": batch_size,
            "num_features": num_features,
            "num_classes": num_classes,
            "lr": lr
        }, output_path)

        print('Model is saved successfully!')
