# By Deqing Song

import os
import numpy as np
import pandas as pd
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics import ConfusionMatrix
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP

import seaborn as sns
import matplotlib.pyplot as plt
import copy

import sys
sys.path.append("Process_Data")
from Data2Models import create_tensor_dataset

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
path_name = os.path.dirname(os.path.abspath(__file__))+'/TrainedModels/'

num_features = 4
num_classes = 4
time_window = 28

batch_size = 64
lr = 0.001
max_epochs = 100  # 训练的最大 epoch 数（因为有 early stopping，可能提前结束）

network_type = 'FCCfC'

# ==========================
# 定义 EarlyStopping 类
# ==========================
class EarlyStopping:
    """基于验证集 Loss 的提前停止."""
    def __init__(self, patience=10, min_delta=1e-4):
        """
        参数:
            patience (int): 如果验证集 Loss 在连续多少个 epoch 内没有提升，则停止。
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

# ==========================
# 定义 RNN 模型
# ==========================
class Sequence(nn.Module):
    def __init__(self, network_type, num_classes=5, num_features=4, time_window=28):
        super(Sequence, self).__init__()
        if network_type == 'LSTM':
            hidden_size = 50
            self.innernet = nn.LSTM(
                input_size=num_features * time_window,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
        elif network_type == 'GRU':
            hidden_size = 50
            self.innernet = nn.GRU(
                input_size=num_features * time_window,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
        elif network_type == 'FCLTC':
            units = 50
            self.innernet = LTC(
                input_size=num_features * time_window,
                units=units,
                batch_first=True
            )
        elif network_type == 'FCCfC':
            units = 50
            self.innernet = CfC(
                input_size=num_features * time_window,
                units=units,
                batch_first=True
            )
        elif network_type == 'NCPLTC':
            units = 53
            input_size = num_features * time_window
            output_size = 50
            self.innernet = LTC(
                input_size=input_size,
                units=AutoNCP(units=units, output_size=output_size),
                batch_first=True
            )
        elif network_type == 'NCPCfC':
            units = 53
            input_size = num_features * time_window
            output_size = 50
            self.innernet = CfC(
                input_size,
                AutoNCP(units=units, output_size=output_size),
                batch_first=True
            )
        self.linear = nn.Linear(in_features=50, out_features=num_classes)
        self.network_type = network_type

    def forward(self, input):
        x, _ = self.innernet(input)
        x = x[:, -1, :]
        x = self.linear(x)
        return x
    
def get_output(data_ds, model):
    """在给定 dataset 上跑推理，返回 (预测标签, 真实标签)。"""
    labels_pred = []
    model.eval()
    with torch.no_grad():
        for i in range(len(data_ds.data_target)):
            x, y = data_ds.__getitem__(i)
            x = x[None, :]  # 扩展 batch 维度 (1, T, Feature)

            logits = model(x)
            logits = logits.squeeze()  # (num_classes,)
            labels_pred.append(logits.detach().numpy())

    labels_pred = np.array(labels_pred)           # shape: (N, num_classes)
    labels_pred = labels_pred.argmax(axis=1)      # 取最大概率对应的类别
    labels_true = np.array(data_ds.data_target[:])
    labels_true = labels_true.astype('int64')

    return torch.tensor(labels_pred), torch.tensor(labels_true)

if __name__ == '__main__':
    # 固定随机种子以保证可复现
    torch.manual_seed(2020)
    np.random.seed(2020)
    random.seed(2020)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        torch.cuda.get_device_name()

    # ==========================
    # 1. 加载数据
    # ==========================
    full_train_data = create_tensor_dataset(
        main_path + 'DATA/6_labeled_window_dataset_train123.csv',
        num_classes=num_classes
    )
    testing_data = create_tensor_dataset(
        main_path + 'DATA/6_labeled_window_dataset_test123.csv',
        num_classes=num_classes
    )

    # ==========================
    # 2. 划分 训练集 / 验证集
    # ==========================
    val_ratio = 0.2  # 验证集占 20%
    train_size = int(len(full_train_data) * (1 - val_ratio))
    val_size = len(full_train_data) - train_size

    train_data, val_data = random_split(
        full_train_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(2020)
    )

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}, Test samples: {len(testing_data)}")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

    # 查看一个 batch 的形状
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # ==========================
    # 3. 构建模型 & 优化器
    # ==========================
    model = Sequence(network_type, num_classes=num_classes).double()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # ==========================
    # 4. Early Stopping 初始化
    # ==========================
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())

    # 用于绘图的记录
    train_loss_history = []
    val_loss_history   = []

    # ==========================
    # 5. 开始训练
    # ==========================
    for epoch in range(max_epochs):
        model.train()
        running_train_loss = []

        # ------ (a) 训练阶段 ------
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss.append(loss.item())

        epoch_train_loss = np.mean(running_train_loss)
        train_loss_history.append(epoch_train_loss)

        # ------ (b) 验证阶段 ------
        model.eval()
        running_val_loss = []
        with torch.no_grad():
            for X_batch, y_batch in val_dataloader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                running_val_loss.append(loss.item())
        epoch_val_loss = np.mean(running_val_loss)
        val_loss_history.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{max_epochs}] - "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}")

        # 检查是否是当前最优
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
        
        # ------ (c) Early Stopping 检查 ------
        if early_stopping(epoch_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}. "
                  f"Best Val Loss: {best_val_loss:.4f}")
            break

    # ==========================
    # 6. 恢复验证集最优权重
    # ==========================
    model.load_state_dict(best_model_weights)
    print("Loaded best model weights (based on val loss).")

    # ==========================
    # 7. 训练 & 验证集 Loss 曲线
    # ==========================
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ==========================
    # 8. 在测试集上评估
    # ==========================
    model.eval()
    confusionMatrixMetric = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    y_pred, y_test = get_output(testing_data, model)
    print('Test set prediction:', y_pred)
    print('Test set ground truth:', y_test)
    print("Confusion Matrix on the test set:\n", confusionMatrixMetric(y_test, y_pred))

    # 画混淆矩阵
    confusionMatrixPlot = confusionMatrixMetric.compute().numpy()
    plt.figure()
    label_classes = ["NC", "ST", "P", "G"]
    sns.heatmap(confusionMatrixPlot,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=label_classes,
                yticklabels=label_classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    plt.show()

    # ==========================
    # 9. 是否保存模型
    # ==========================
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
            "n_epochs": epoch + 1,  # 真实训练到第几个 epoch
            "batch_size": batch_size,
            "num_features": num_features,
            "num_classes": num_classes,
            "lr": lr
        }, output_path)
        
        print('model is saved successfully!')
