"""
通过用户输入手势名称并录入手势数据，进行特征提取并存储到手势字典中

"""

import numpy as np
import os

import torch.nn as nn
import torch.optim as optim

import DataProcess as DP

# define hyperparameters for training data
global window_size, step_size
window_size = 28
step_size = 14
flatten_mode = "flatten"

def reshape_data(data, flatten_mode="flatten"):
    """
    根据 flatten_mode 参数选择不同的数据展平或矩阵保留方式。
    参数:
        data: 输入的原始数据 (28, 4, 7) 格式
        flatten_mode: 选择展平方式的字符串
                      "flatten" - 完全展平为 1D
                      "28x4x7" - 保持原始三维格式
                      "7x(4x28)" - 7 行，每行包含 4x28 特征的 2D 矩阵
                      "4x(7x28)" - 4 行，每行包含 7x28 特征的 2D 矩阵
    返回值:
        处理后的数据
    """
    if flatten_mode == "flatten":
        # 完全展平
        return data.flatten()  # 1D array, shape (784,)
    elif flatten_mode == "28x4x7":
        # 保持原始三维结构
        return data  # 3D array, shape (28, 4, 7)
    elif flatten_mode == "7x(4x28)":
        # 7 个关节点，每个关节一个 (4*28) 的特征向量
        return data.reshape(7, 4 * 28)  # 2D array, shape (7, 112)  
    elif flatten_mode == "4x(7x28)":
        # 4 个特征，每个特征是一个 7x28 的矩阵
        return data.reshape(4, 7 * 28)  # 2D array, shape (4, 196)
    else:
        raise ValueError("Invalid flatten_mode. Choose from 'flatten', '28x4x7', '7x(4x28)', '4x(7x28)'.")




class User(object):
    def __init__(self, user_name:str, password='NotSet', storage_dir='user_data'): # 这里的路径和add_gesture的路径可以考虑怎么intergate一下
        self.user_name = user_name
        self.password = password

        self.gesture_dict = {}
        self.gesture_num = len(self.gesture_dict)

        self.gesture_storage_dir = os.path.join(storage_dir, self.user_name)
        os.makedirs(self.gesture_storage_dir, exist_ok=True)

    def get_gesture_info(self):
        gesture_list = list(self.gesture_dict.keys())
        print(f"Gesture list: {gesture_list}")
        print(f"In total {self.gesture_num} gestures")

    def add_gesture(self, gesture_name, gesture_data_raw_dir, gesture_data_process_save_dir):
        if gesture_name in self.gesture_dict:
            print(f"Gesture '{gesture_name}' already exists for user '{self.user_name}'.") 
        else:
            gesture_label_data = DP.labelData(gesture_name, gesture_data_raw_dir, gesture_data_process_save_dir)
            gesture_window_data = DP.windowData(gesture_label_data, gesture_data_process_save_dir, window_size, step_size)
            gesture = Gesture(gesture_name=gesture_name, gesture_training_data=gesture_window_data)
            self.gesture_dict[gesture_name] = gesture
            print(f"Gesture '{gesture_name}' added for user '{self.user_name}'.")

# 收集手势数据并提取相关特征
class Gesture(object):
    # 初始化：收集手势数据，
    def __init__(self, gesture_name:str, gesture_data):
        self.gesture_name = gesture_name    
        self.gesture_data = gesture_data # is a dataframe for labeled_window_data
        ## TODO: here initiall the model, train it and save it
        self.gesture_model = RBFNetwork(num_centers, num_classes) # 参数还没填写
        RBFNetwork.train()
        self.model_criterion = nn.CrossEntropyLoss()
        self.model_optimizer = optim.Adam(self.gesture_model.parameters(), lr=0.001)
        
    
    def classifier_train(self):
        # 使用self.gesture_data访问输入数据 其实还没想好数据该在哪一步进来
        # 使用self.gesture_model实例化网络模型
    

class RBFNetwork:
    def __init__(self, rho=0.7, epsilon=0.5, v0=0.1):
        # 初始化RBF网络参数
        self.rho = rho       # 输入相似度阈值
        self.epsilon = epsilon  # 输出相似度阈值
        self.v0 = v0         # 初始偏差值
        self.centers = []    # 存储聚类中心
        self.variances = []  # 存储每个聚类的方差
        self.weights = None  # 存储输出层的权重

    def _input_similarity(self, x, center, variance):
        # 计算输入样本x与聚类中心的相似度
        similarity = np.exp(-np.sum(((x - center) / variance) ** 2))
        # print(f"_input_similarity -> x: {x}, center: {center}, variance: {variance}, similarity: {similarity}")
        return similarity

    def _output_similarity(self, y, cluster_y):
        # 计算输出样本y与聚类输出的相似度
        similarity = np.sum(np.minimum(y, cluster_y)) / np.sum(np.maximum(y, cluster_y))
        # print(f"_output_similarity -> y: {y}, cluster_y: {cluster_y}, similarity: {similarity}")
        return similarity

    def _add_cluster(self, x, y):
        # 如果没有找到合适的聚类，则创建新聚类
        self.centers.append(x)  # 新聚类中心为当前样本
        self.variances.append(np.full(x.shape, self.v0))  # 初始化方差
        # print(f"_add_cluster -> New center added: {x}, variance: {self.variances[-1]}")
        return y

    def _update_cluster(self, x, y, cluster_idx):
        # 更新已有聚类的中心和方差
        n = len(self.centers[cluster_idx])  # 当前聚类中样本数量
        old_center = self.centers[cluster_idx].copy()
        old_variance = self.variances[cluster_idx].copy()
        
        # 更新聚类中心
        self.centers[cluster_idx] = (n * self.centers[cluster_idx] + x) / (n + 1)
        # 更新方差
        self.variances[cluster_idx] = np.sqrt(((self.variances[cluster_idx]**2) * n + (x - self.centers[cluster_idx])**2) / (n + 1))
        
        # print(f"_update_cluster -> Old center: {old_center}, New center: {self.centers[cluster_idx]}")
        # print(f"_update_cluster -> Old variance: {old_variance}, New variance: {self.variances[cluster_idx]}")

    def fit(self, X, y):
        # 训练RBF网络
        cluster_labels = []
        for i, x in enumerate(X):
            max_input_sim, best_cluster = -1, -1
            # print(f"fit -> Processing sample {i}, x: {x}, y: {y[i]}")
            
            # 查找最佳聚类
            for idx, center in enumerate(self.centers):
                input_sim = self._input_similarity(x, center, self.variances[idx])  # 输入相似度
                output_sim = self._output_similarity(y[i], cluster_labels[idx])  # 输出相似度
                # 如果相似度超过阈值，则更新最佳聚类
                # print(f"fit -> Cluster {idx}: input_sim: {input_sim}, output_sim: {output_sim}")
                if input_sim >= self.rho and output_sim >= self.epsilon:
                    if input_sim > max_input_sim:
                        max_input_sim, best_cluster = input_sim, idx
                        # print(f"fit -> Updated best cluster to {best_cluster} with input_sim: {max_input_sim}")

            # 如果没有找到合适的聚类，则创建新聚类
            if best_cluster == -1:
                cluster_labels.append(self._add_cluster(x, y[i]))
                # print(f"fit -> Created new cluster for sample {i}")
            else:
                self._update_cluster(x, y[i], best_cluster)
                # print(f"fit -> Updated cluster {best_cluster} for sample {i}")

        # 计算隐藏层输出并优化权重
        # print("--------Calculating hidden layer output--------")
        hidden_layer_output = np.array([
            [self._input_similarity(x, center, var) for center, var in zip(self.centers, self.variances)]
            for x in X
        ])
        # print(f"fit -> Hidden layer output matrix:\n{hidden_layer_output}")
        
        # 使用伪逆计算输出层的权重
        self.weights = np.linalg.pinv(hidden_layer_output) @ y
        # print(f"fit -> Calculated weights: {self.weights}")

    def predict(self, X):
        # 使用训练好的模型进行预测
        hidden_layer_output = np.array([
            [self._input_similarity(x, center, var) for center, var in zip(self.centers, self.variances)]
            for x in X
        ])
        # print(f"predict -> Hidden layer output for predictions:\n{hidden_layer_output}")
        
        # 输出大于0.5的预测为10（YES），否则为-1（NO）
        predictions = np.where(hidden_layer_output @ self.weights >= 0.5, 10, -1)
        # print(f"predict -> Predictions: {predictions}")
        return predictions

    def get_params(self, deep=True):
        return {"rho": self.rho, "epsilon": self.epsilon, "v0": self.v0}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
   

