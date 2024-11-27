"""
通过用户输入手势名称并录入手势数据，进行特征提取并存储到手势字典中

"""

from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import numpy as np

import DataReader as DP

# define hyperparameters for training data
global window_size, step_size
window_size = 28
step_size = 14
flatten_mode = "flatten"
num_features = 4
dof = 7

# rho=1.9
epsilon=0
# v0=1/np.sqrt(2)

def preprocess_data(window_df, flatten_mode):
    """
    根据 flatten_mode 参数选择不同的数据展平或矩阵保留方式。
    参数:
        data: 输入的原始数据 (28, 4, 7) 格式
        flatten_mode: 选择展平方式的字符串
                      "flatten" - 完全展平为 1D
                      "28x4x7" - 保持原始三维格式
                      "7x(4x28)" - 7 行，每行包含 4x28 特征的 2D 矩阵
    返回值:
        处理后的数据
    """
    joint0_colums = ['e0','de0','tau_J0','tau_ext0']
    joint1_colums = ['e1','de1','tau_J1','tau_ext1']
    joint2_colums = ['e2','de2','tau_J2','tau_ext2']
    joint3_colums = ['e3','de3','tau_J3','tau_ext3']
    joint4_colums = ['e4','de4','tau_J4','tau_ext4']
    joint5_colums = ['e5','de5','tau_J5','tau_ext5']
    joint6_colums = ['e6','de6','tau_J6','tau_ext6']

    joints_colums = [joint0_colums, joint1_colums, joint2_colums, joint3_colums, joint4_colums, joint5_colums, joint6_colums]

    # print(window_df)
    if flatten_mode == "flatten":
        # 完全展平(4*7*28)
        flat_joints_colums = sum(joints_colums,[])
        data = window_df.loc[:, flat_joints_colums].values.flatten()
        return np.array(data)  # 1D array, shape (784,)
    elif flatten_mode == "28x4x7":
        flat_joints_colums = sum(joints_colums,[])
        data_i = window_df[flat_joints_colums].values
        data = data_i.reshape(-1, dof, num_features)
        return np.array(data)  # 3D array, shape (28, 4, 7)
    elif flatten_mode == "7x(4*28)":
        # 7 个关节点，每个关节一个 (4*28) 的特征向量
        data = np.zeros((dof,num_features * window_df.shape[0])) # generate a initial numpy array 7x(4*28)
        for i, joint_colums in enumerate(joints_colums):
            data_i = window_df.loc[:, joint_colums].values.flatten()
            data[i,:] = data_i
        return np.array(data) # 2D array, shape (7, 112)  
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
        save = ''
        if gesture_name in self.gesture_dict:
            print(f"Gesture '{gesture_name}' already exists for user '{self.user_name}'.") 
        else:
            gesture_label_data = DP.labelData(gesture_name, gesture_data_raw_dir, gesture_data_process_save_dir)
            gesture_window_data = DP.windowData(gesture_label_data, gesture_data_process_save_dir, window_size, step_size)
            gesture = Gesture(gesture_name=gesture_name, gesture_data=gesture_window_data)
            self.gesture_dict[gesture_name] = gesture
            save = input('Do you wish to save this gesture?(Y/n): ')
            if save == 'Y':
                print()
                with open(os.path.join(self.gesture_storage_dir, f'{gesture_name}.pickle'),"wb") as file:
                    pickle.dump(gesture,file)
                print(f"Gesture '{gesture_name}' for user '{self.user_name}' is saved.")
            print(f"Gesture '{gesture_name}' added for user '{self.user_name}'.")

# 收集手势数据并提取相关特征
class Gesture(object):
    # 初始化：收集手势数据，
    def __init__(self, gesture_name:str, gesture_data):
        self.gesture_name = gesture_name    
        self.gesture_data = gesture_data # is a dataframe for labeled_window_data
        # here initiall the model, train it and save it
        self.gesture_model = RBFNetwork(epsilon=epsilon) 
        self._data_split()
        # print(f"this is befor init best_rho {best_rho}")

        # best_rho = self.gesture_model.cross_validate(self.X_train, self.y_train)
        # print(f"this is init best_rho {best_rho}")
        # self.gesture_model.rho = best_rho 

        self.classifier_train()
        self.classifier_test()
        print(f'Gesture {self.gesture_name} is established!')
        
    def _data_split(self):
        # 根据 block_id 进行分组
        grouped = self.gesture_data.groupby('window_id')
        # 将分组后的数据块列表化
        windows = [group.reset_index(drop=True) for _, group in grouped]
        train_windows, test_windows = train_test_split(windows, test_size=0.2, random_state=42)
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for window in train_windows:
            # print(window)
            preprocessed_window = preprocess_data(window_df = window.drop(columns=['label_idx','label_name','window_id','window_gesture_idx', 'window_gesture_name']),
                                                  flatten_mode = flatten_mode)
            X_train.append(preprocessed_window)
            y_train.append(window.loc[0,'window_gesture_idx'])
        for window in test_windows:
            preprocessed_window = preprocess_data(window_df = window.drop(columns=['label_idx','label_name','window_id','window_gesture_idx', 'window_gesture_name']),
                                                  flatten_mode = flatten_mode)
            X_test.append(preprocessed_window)
            y_test.append(window.loc[0,'window_gesture_idx'])
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        # print(f"y_train is {self.y_train}")
        # print(type(self.X_train[0][0]))
        # print(type(self.y_train[0]))
        # print(f"y_test is {self.y_test}")
        
    def classifier_train(self):
        self.gesture_model.fit(self.X_train, self.y_train)

    def classifier_test(self):
        self.y_predict,self.y_percent = self.gesture_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, self.y_predict)

        print(f"y_predictions is {len(self.y_predict)}")
        print(f"y_predictions is {self.y_predict}")
        print(f"y_percent is {len(self.y_percent)}")
        print(f"y_percent is {self.y_percent}")
        print(f"y_test are {len(self.y_test)}")
        print(f"y_test are {self.y_test}")
        print("Test Results:", ["YES" if pred == 1 else "NO" for pred in self.y_predict])
        print(f"Accuracy: {accuracy:.2f}")
    

class RBFNetwork(object):
    def __init__(self,epsilon=0.5, v0=None):
        # print("DEBUG: this is init")
        # 初始化RBF网络参数
        # self.rho = None      # 输入相似度阈值
        self.epsilon = epsilon  # 输出相似度阈值
        self.v0 = v0 if v0 is not None else 0
        self.centers = []    # 存储聚类中心
        self.variances = []  # 存储每个聚类的方差
        self.weights = None  # 存储输出层的权重
        self.cluster_labels = []

    def set_v0(self, X):
        """基于输入数据X自动计算v0值"""
        # 计算X每个维度的方差
        feature_variances = np.var(X, axis=0)  # 计算每个特征的方差
        # feature_variances = np.std(X, axis=0) # 计算每个特征的标准差
        # print(f"feature_variances is {feature_variances}")
        # feature_variances =
        self.v0 = np.max(feature_variances)  # 使用方差的均值作为v0的初始值
        # self.v0 = 1/np.sqrt(2)
        # self.v0=50000
        # print(f"Dynamic v0 set to: {self.v0}")  # 输出动态设置的 v08    

    def _input_similarity(self, x, center,variance):
        # print("DEBUG: this is _input_similarity")
        # 计算输入样本x与聚类中心的相似度
        # print(-np.linalg.norm(x - center))
        similarity = np.exp(-np.linalg.norm(x - center) ** 2 / (2 * variance ** 2))
        # similarity = np.corrcoef(x,center)[0,1]
        # print(f"_input_similarity -> x: {x}, center: {center}, variance :{variance} ,similarity: {similarity}")
        return similarity

    def _output_similarity(self, y, cluster_y):
        # print("DEBUG: this is _output_similarity")
        # 计算输出样本y与聚类输出的相似度
        similarity = np.sum(np.minimum(y, cluster_y)) / np.sum(np.maximum(y, cluster_y))
        # print(f"_output_similarity -> y: {y}, cluster_y: {cluster_y}, similarity: {similarity}")
        return similarity

    def _add_cluster(self, x, y):
        # print("DEBUG: this is _add_cluster")
        # 如果没有找到合适的聚类，则创建新聚类
        self.centers.append(x)  # 新聚类中心为当前样本
        self.variances.append(self.v0)  # 初始化方差
        self.cluster_labels.append(y)
        # print(f"_add_cluster -> New center added: {x}, variance: {self.variances[-1]}")
        return y

    def _update_cluster(self, x, y, cluster_idx):
        # print("DEBUG: this is _update_cluster")
        # 更新已有聚类的中心和方差
        n = len(self.centers[cluster_idx])  # 当前聚类中样本数量
        old_center = self.centers[cluster_idx]
        old_variance = self.variances[cluster_idx]
        
        # 更新聚类中心
        self.centers[cluster_idx] = (n * self.centers[cluster_idx] + x) / (n + 1)
        # 更新方差
        self.variances[cluster_idx] = np.sqrt(((self.variances[cluster_idx]**2) * n + np.linalg.norm(x - self.centers[cluster_idx])**2) / (n + 1))
        
        # print(f"_update_cluster -> Old center: {old_center}, New center: {self.centers[cluster_idx]}")
        # print(f"_update_cluster -> Old variance: {old_variance}, New variance: {self.variances[cluster_idx]}")

    def compute_similarity_scores(self, X):
        # Compute pairwise similarity scores for the data points in X
        similarity_matrix = np.zeros((len(X), len(X)))
        
        for i in range(len(X)):
            for j in range(i, len(X)):
                similarity_matrix[i, j] = self._input_similarity(X[i], X[j], self.v0)
                similarity_matrix[j, i] = similarity_matrix[i, j]  # Symmetric matrix
        # print(similarity_matrix)
        return similarity_matrix

    def determine_rho_range(self,X):
        # print("DEBUG: this is determine_rho_range")
        similarity_matrix = self.compute_similarity_scores(X)
        # Flatten the upper triangular part of the matrix to avoid duplicate comparisons
        upper_triangular = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        # print(upper_triangular)
        # Calculate the 10th and 90th percentiles of the similarity scores
        rho_min = np.percentile(upper_triangular, 10)
        rho_max = np.percentile(upper_triangular, 90)
        # print(max(upper_triangular))
        # print(min(upper_triangular))
        # print(rho_max)
        # print(rho_min)
        return rho_min, rho_max

    def reset_params(self):
        self.v0 = 0        # 初始偏差值
        self.centers = []    # 存储聚类中心
        self.variances = []  # 存储每个聚类的方差
        self.weights = None  # 存储输出层的权重
        self.cluster_labels = [] 

    def fit(self, X, y):
        # print("DEBUG: this is fit")
        self.set_v0(X)

        # Determine range for rho and create grid of rho values
        rho_min, rho_max = self.determine_rho_range(X)
        rho_values = np.linspace(rho_min, rho_max, num=10)
        # rho_values = [(rho_min+rho_max)/2]
        best_rho = None
        best_accuracy = 0

        # Grid search over rho values
        for rho in rho_values:
            self.rho = rho
            # print(f'rho = {self.rho}')
            fold_accuracies = []
            
            # Train and validate model on each rho
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            for fold_num, (train_index, val_index) in enumerate(kf.split(X)):
                # print(f"Fold {fold_num+1}:")
                # print(f"train_index: {train_index}")
                # print(f"val_index: {val_index}")
                
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # Reset model parameters before each fold
                self.reset_params()
                self.set_v0(X) # also can be set to X_train
                
                self.fit_clusters(X_train, y_train)
                y_pred = self.predict(X_val)[0]
                accuracy = accuracy_score(y_val, y_pred)
                fold_accuracies.append(accuracy)
                # print(f'Accuracy: {accuracy}' )
                
            avg_accuracy = np.mean(fold_accuracies)
            # print(f'Ave Accuracy: {avg_accuracy}')
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_rho = rho

        self.rho = best_rho
        # print(f"Selected best rho: {self.rho}")

        # Final model training with the best rho
        self.fit_clusters(X, y)

    def fit_clusters(self, X, y):
        # print(f"Debug: this is fit_clusters")
        # 训练RBF网络
        # self.rho = self.cross_validate(X, y)  # Set the best rho
        

        if not self.centers:
            self.centers.append(X[0])
            self.variances.append(self.v0)
            self.cluster_labels.append(y[0])

        for i, x in enumerate(X):
            max_input_sim, best_cluster = -1, -1
            # print(f"fit -> Processing sample {i}, x: {x}, y: {y[i]}")
            
            # 查找最佳聚类
            for idx, center in enumerate(self.centers):
                # print(f"debug: the length of centers is {len(self.centers)}")
                # print(f"debug: the length of cluster_labels is {len(self.cluster_labels)}")
                # print(f"debug: the idex  is {idx} ")
                input_sim = self._input_similarity(x, center, self.variances[idx])  # 输入相似度
                output_sim = self._output_similarity(y[i], self.cluster_labels[idx])  # 输出相似度
                # best_rho = self.cross_validate(X,y,k=5)
                # 如果相似度超过阈值，则更新最佳聚类
                # print(f"fit -> Cluster {idx}: input_sim: {input_sim}, output_sim: {output_sim}")
                # print(f"in the fit(), the  is {best_rho}")
                if input_sim >= self.rho and output_sim >= self.epsilon:
                    if input_sim > max_input_sim:
                        max_input_sim, best_cluster = input_sim, idx
                        # print(f"fit -> Updated best cluster to {best_cluster} with input_sim: {max_input_sim}")

            # 如果没有找到合适的聚类，则创建新聚类
            if best_cluster == -1:
                self._add_cluster(x, y[i])
                # print(f"fit -> Created new cluster for sample {i}")
            else:
                self._update_cluster(x, y[i], best_cluster)
                # print(f"fit -> Updated cluster {best_cluster} for sample {i}")

        # 计算隐藏层输出并优化权重
        # print("--------Calculating hidden layer output--------")
        hidden_layer_output = np.array([
            [self._input_similarity(x, center,variance) for center ,variance,in zip(self.centers,self.variances)]
            for x in X
        ])
        # print(f"fit -> Hidden layer output matrix:\n{hidden_layer_output}")
        
        # 使用伪逆计算输出层的权重
        self.weights = np.linalg.pinv(hidden_layer_output) @ y
        # print(f"fit -> Calculated weights: {self.weights}")

    def softmax(self,x):
        exp_x = np.exp(x)
        return exp_x/np.sum(exp_x)

    def predict(self, X):
        # print(f"Debug: this is predict")
        # 使用训练好的模型进行预测
        hidden_layer_output = np.array([
            [self._input_similarity(x, center,variance) for center ,variance,in zip(self.centers,self.variances)]
            for x in X
        ])
        raw_output = hidden_layer_output @ self.weights
        # print(f"predict -> Hidden layer output for predictions:\n{hidden_layer_output}")
        # print(f"{prt}]")
        """ plt.hist(prt,bins=10,color='red',alpha=0.6,label = 'prt distribution')
        plt.show() """
        # 输出大于0.5的预测为1（YES），否则为-1（NO）
        # print(hidden_layer_output @ self.weights)
        predictions = np.where(hidden_layer_output @ self.weights >= 0, 1, -1)
        # print(f"predict -> Predictions: {predictions}")
        percentages = self.softmax(raw_output)
        return predictions,percentages
    
    def single_predict(self, x):
        print(f"Debug: this is single_predict")
       
        # 使用训练好的模型进行预测
        hidden_layer_output = [self._input_similarity(x, center,variance) for center ,variance,in zip(self.centers,self.variances)]
        # print(f"predict -> Hidden layer output for predictions:\n{hidden_layer_output}")
        # 输出大于0.5的预测为1（YES），否则为-1（NO）
        raw_output = hidden_layer_output @ self.weights
        prediction = 1 if raw_output >= 0 else -1
        # print(f"predict -> Predictions: {predictions}")
        percentage = self.softmax(raw_output)
        # print(f"percentages -> percentages: {percentages}")
        return prediction,percentage
   

if __name__ == '__main__':
    UName = 'TestU1'
    UPass = '123465'
    CurrentUser = User(user_name=UName, password=UPass)
    CurrentUser.get_gesture_info
    # add_gesture_flag = input('Add new gesture?(Y/n):')
    add_gesture_flag = 'Y'
    if add_gesture_flag == 'Y':
        GName = 'TestG1'
        # Data Recording Progress should be added here
        CurrentUser.add_gesture(GName, 'DATA/rawData/0910-7ST-S1', CurrentUser.gesture_storage_dir)
