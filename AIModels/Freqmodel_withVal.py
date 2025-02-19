import numpy as np
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from sklearn.model_selection import train_test_split
import copy

import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append("Process_Data")
from Data2Models import create_tensor_dataset_stft

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
path_name = os.path.dirname(os.path.abspath(__file__))+'/TrainedModels/'

num_classes = 4
time_window = 28

batch_size = 64
lr = 0.001
max_epochs = 100  # 最大训练轮数，Early Stopping 可能会提早终止
network_type = 'T2L3DCNN'

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
# 2) 定义 CNN 模型结构
# ==================================================
class CNNSequence(nn.Module):
    def __init__(self, network_type, num_classes):
        super(CNNSequence, self).__init__()

        if network_type == '2LCNN':
            self.conv1 = nn.Conv2d(in_channels=28, out_channels=16, kernel_size=3, stride=1, padding=0)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(32 * 5 * 5, 64)
            self.fc2 = nn.Linear(64, num_classes)

        elif network_type == '3LCNN':
            self.conv1 = nn.Conv2d(in_channels=28, out_channels=16, kernel_size=3, stride=1, padding=0)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64* 14 * 9, 128)
            self.fc2 = nn.Linear(128, num_classes)

        self.network_type = network_type
        self.num_classes = num_classes

    def forward(self, x):
        if self.network_type == '2LCNN':
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.max_pool2d(x, (1, 2))
            x = nn.functional.relu(self.conv2(x))
            x = nn.functional.max_pool2d(x, (1, 2))
            x = self.flatten(x)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.network_type == '3LCNN':
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.avg_pool2d(x, (2, 2))
            x = nn.functional.relu(self.conv2(x))
            x = nn.functional.avg_pool2d(x, (2, 1))
            x = nn.functional.relu(self.conv3(x))
            x = nn.functional.avg_pool2d(x, (2, 1))
            x = self.flatten(x)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)

        return x
    
class CNNSequence3D(nn.Module):
    def __init__(self, network_type, num_classes):
        super(CNNSequence3D, self).__init__()

        if network_type in ['2L3DCNN', 'T2L3DCNN']:
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=16,
                                   kernel_size=(28, 3, 3), stride=1, padding=0)
            self.conv2 = nn.Conv3d(in_channels=16, out_channels=32,
                                   kernel_size=(1, 3, 3), stride=1, padding=0)
            self.global_max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(32, num_classes)
        
        self.network_type = network_type
        self.num_classes = num_classes

    def forward(self, input):
        if self.network_type in ['2L3DCNN', 'T2L3DCNN']:
            x = input.unsqueeze(1)
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.relu(self.conv2(x))
            x = self.global_max_pool(x)
            x = self.flatten(x)
            x = self.fc(x)
        return x

# ==================================================
# 3) 推理函数：在给定 dataset 上跑推理
# ==================================================
def get_output(data_ds, model):
    labels_pred = []
    model.eval()
    with torch.no_grad():
        for i in range(len(data_ds.labels)):
            x, y = data_ds.__getitem__(i)
            x = x.unsqueeze(0)  # (1, C, H, W) 或 (1, 1, D, H, W)
            logits = model(x)
            logits = logits.squeeze()
            labels_pred.append(logits.detach().numpy())
  
    labels_pred = np.array(labels_pred)           # shape: (N, num_classes)
    labels_pred = labels_pred.argmax(axis=1)      # 取最大概率对应的类别
    labels_true = np.array(data_ds.labels[:])
    labels_true = labels_true.astype('int64')
    return torch.tensor(labels_pred), torch.tensor(labels_true)

# ==================================================
# 4) 主函数
# ==================================================
if __name__ == '__main__':
    # 固定随机种子
    torch.manual_seed(2024)
    np.random.seed(2024)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------
    # 4.1) 加载原始数据 & 初步 train/test 划分
    # -----------------------------
    str2int_4 = {'NC': 0, 'ST': 1, 'P': 2, 'G': 3}
    if network_type in ['2LCNN', '2L3DCNN']:
        loaded_data = np.load('DATA/STFT_images/stft_matrices_123.npz', allow_pickle=True)
        all_matrices = loaded_data['stft_matrices']
        labels_str   = loaded_data['labels']
        labels       = [str2int_4[s] for s in labels_str]
    elif network_type == '3LCNN':
        loaded_data = np.load('DATA/CWT_images/cwt_matrices.npz', allow_pickle=True)
        all_matrices = loaded_data['cwt_matrices']
        labels_str   = loaded_data['labels']
        labels       = [str2int_4[s] for s in labels_str]
    elif network_type == 'T2L3DCNN':
        loaded_data  = np.load('DATA/T_images/T_matrices_123.npz', allow_pickle=True)
        all_matrices = loaded_data['T_matrices']
        labels_str   = loaded_data['labels']
        labels       = [str2int_4[s] for s in labels_str]
    else:
        raise ValueError("Unsupported network_type")

    # 先从全部数据中抽 20% 当做测试集
    train_matrices, test_matrices, train_labels, test_labels = \
        train_test_split(all_matrices, labels, test_size=0.2, random_state=2024)

    # -----------------------------
    # 4.2) 从 train_matrices 中再划分验证集
    # -----------------------------
    train_matrices2, val_matrices, train_labels2, val_labels = \
        train_test_split(train_matrices, train_labels, test_size=0.2, random_state=2025)
    print(f"Train samples: {len(train_matrices2)}, "
          f"Val samples: {len(val_matrices)}, "
          f"Test samples: {len(test_matrices)}")

    # -----------------------------
    # 4.3) 构建 DataSet & DataLoader
    # -----------------------------
    training_data = create_tensor_dataset_stft(stft_matrices=train_matrices2, labels=train_labels2)
    val_data      = create_tensor_dataset_stft(stft_matrices=val_matrices,   labels=val_labels)
    testing_data  = create_tensor_dataset_stft(stft_matrices=test_matrices,  labels=test_labels)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_data,      batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(testing_data,  batch_size=batch_size, shuffle=False)

    # 查看一个 batch 的形状
    train_features, train_labels_example = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels_example.size()}")

    # -----------------------------
    # 4.4) 构建模型 & 优化器
    # -----------------------------
    if network_type in ['2LCNN', '3LCNN']:
        model = CNNSequence(network_type=network_type, num_classes=num_classes)
    else:  # '2L3DCNN', 'T2L3DCNN'
        model = CNNSequence3D(network_type=network_type, num_classes=num_classes)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # -----------------------------
    # 4.5) Early Stopping 初始化
    # -----------------------------
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    best_val_loss = float('inf')
    best_weights  = copy.deepcopy(model.state_dict())

    # 用于绘图的训练/验证 Loss
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
            loss   = loss_fn(y_pred, y_batch)
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
                loss   = loss_fn(y_pred, y_batch)
                running_val_loss.append(loss.item())

        epoch_val_loss = np.mean(running_val_loss)
        val_loss_history.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{max_epochs}] - "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}")

        # 记录验证集最优权重
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights  = copy.deepcopy(model.state_dict())

        # ---- (c) Early Stopping 检查 ----
        if early_stopping(epoch_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}. "
                  f"Best Val Loss: {best_val_loss:.4f}")
            break

    # -----------------------------
    # 4.7) 恢复验证集最优权重
    # -----------------------------
    model.load_state_dict(best_weights)
    print("Loaded best weights (based on val loss).")

    # -----------------------------
    # 4.8) Loss 曲线可视化
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
    # 4.9) 在测试集上评估
    # -----------------------------
    model.eval()
    confusionMatrixMetric = ConfusionMatrix(task='multiclass', num_classes=num_classes)

    # 测试集
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred_logits = model(X_batch)
            preds = y_pred_logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y_batch.cpu())
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # 打印混淆矩阵
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
            "n_epochs": epoch+1,  # 训练到第几个epoch
            "batch_size": batch_size,
            "num_classes": num_classes,
            "lr": lr
        }, output_path)

        print('Model is saved successfully!')
