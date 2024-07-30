import numpy as np
import torch
import torch.nn as nn

# 定义一个继承自 nn.Module 的类 Sequence
class Sequence(nn.Module):
    # 初始化函数，定义类的属性和层
    def __init__(self, num_class=5, network_type='main', num_features_lstm=4):
        # 调用父类 nn.Module 的初始化方法
        super(Sequence, self).__init__()
        # 定义 LSTM 层，输入特征维度为 num_features_lstm*28，隐藏层大小为 hidden_size
        hidden_size = 50
        self.lstm = nn.LSTM(input_size=num_features_lstm*28, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.network_type = network_type
        # 根据 network_type 决定线性层的输入大小
        if self.network_type == 'main':
            self.linear = nn.Linear(hidden_size, num_class)
        else:
            self.linear = nn.Linear(hidden_size*7, num_class)

    # 定义前向传播函数
    def forward(self, input, future=0):
        # 将输入数据传入 LSTM 层
        x, _ = self.lstm(input)
        # 如果网络类型是 'main'，取 LSTM 输出的最后一个时间步
        if self.network_type == 'main':
            x = x[:, -1, :]
        else:
            # 否则将 LSTM 的输出展平成一个长向量
            x = torch.flatten(x, start_dim=1)
        # 通过线性层进行预测
        x = self.linear(x)
        return x

# 获取模型输出函数
def get_output(data_ds, model):
    labels_pred = []
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        for i in range(data_ds.data_target.shape[0]):
            # 获取数据集中的样本和标签
            x, y = data_ds.__getitem__(i)
            x = x[None, :]  # 添加批次维度
            x = model(x)  # 通过模型进行预测
            x = x.squeeze()  # 去除多余的维度
            labels_pred.append(x.detach().numpy())  # 将预测结果添加到列表中
    # 将预测结果列表转换为数组
    labels_pred = np.array(labels_pred)
    # 获取每个样本的预测类别
    labels_pred = labels_pred.argmax(axis=1)
    # 获取真实标签
    labels_true = np.array(data_ds.data_target[:])
    labels_true = labels_true.astype('int64')
    # 返回预测标签和真实标签的张量
    return torch.tensor(labels_pred), torch.tensor(labels_true)

# 导入 LSTM 模型的函数
def import_lstm_models(PATH: str, num_features_lstm=4):
    # 加载模型检查点
    checkpoint = torch.load(PATH)
    # 根据检查点信息初始化模型
    model = Sequence(num_class=checkpoint["num_classes"], network_type=checkpoint["network_type"], num_features_lstm=num_features_lstm)
    # 加载模型状态字典
    model.load_state_dict(checkpoint["model_state_dict"])

    # 根据检查点信息设置标签映射字典
    if checkpoint["collision"]:
        labels_map = {
            0: ',Collaborative_Contact',
            1: ',Collision,'
        }
        print('collision detection model is loaded!')
    elif checkpoint["localization"]:
        labels_map = {
            0: ',Link 5',
            1: ',Link 6,'
        }
        print('localization model is loaded!')
    elif checkpoint["num_classes"] == 5:
        labels_map = {
            0: ',Noncontact,',
            1: ',Intentional_Link5,',
            2: ',Intentional_Link6,',
            3: ',Collision_Link5,',
            4: ',Collision_Link6,',
        }
        print('5-classes model is loaded!')
    elif checkpoint["num_classes"] == 3:
        labels_map = {
            0: ',Noncontact,',
            1: ',Collaborative_Contact,',
            2: ',Collision,',
        }
        print('collision detection with 3 classes model is loaded!')
    elif checkpoint["num_classes"] == 2:
        labels_map = {
            0: ',Noncontact,',
            1: ',Contact,',
        }
        print('contact detection model is loaded!')

    # 返回评估模式下的模型和标签映射字典
    return model.eval(), labels_map

# 导入旧版本 LSTM 模型的函数
def import_lstm_models_old(PATH: str, num_classes: int, network_type: str, model_name: str):
    # 初始化模型
    model = Sequence(num_class=num_classes, network_type=network_type)
    # 加载模型检查点
    checkpoint = torch.load(PATH + model_name)
    # 加载模型状态字典
    model.load_state_dict(checkpoint["model_state_dict"])
    print('***  Models loaded  ***')
    # 返回评估模式下的模型
    return model.eval()
