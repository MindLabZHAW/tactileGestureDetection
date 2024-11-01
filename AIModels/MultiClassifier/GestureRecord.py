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
    """
    RBFNetwork is designed to classify gestures by comparing new gestures to known gesture patterns.
    It uses Radial Basis Functions (RBFs) as hidden units that calculate the similarity between inputs
    and learned centers (prototypes). This binary classification network outputs YES or NO depending on
    whether the new gesture matches a known gesture pattern.
    
    Parameters:
        num_centers (int): Number of RBF centers to use, representing prototypes of each gesture class.
        num_classes (int): Number of output classes (2 for binary classification: YES/NO).
        flatten_mode (str): Mode to preprocess input data, flattening or reshaping as specified.
    """

    def __init__(self,num_centers,num_classes,flatten_mode = "flatten"):
        """
        Initializes the RBFNetwork, setting the number of centers and classes.
        Also initializes parameters for the centers, sigma values, and weights, which will be learned.
        
        Parameters:
            num_centers (int): Number of RBF centers, allowing the network to capture intra-class variability.
            num_classes (int): Number of output classes (2 for binary classification).
            flatten_mode (str): Specifies how input data will be preprocessed for the network.
        """
        self.num_centers = num_centers
        self.num_classes = num_classes
        self.flatten_mode = flatten_mode
        self.num_features = None
        self.centers = None
        self.sigmas = np.ones(num_centers)  # 初始化标准差
        self.weights = None


    def preprocess_input(self, data):
        """
        Preprocesses the input data based on the specified flatten_mode.
        
        Parameters:
            data (ndarray): Input gesture data in original shape.
        
        Returns:
            ndarray: Processed data in the specified format (e.g., flattened to 1D).
        """
        processed_data = reshape_data(data, flatten_mode=self.flatten_mode)
        self.num_features = processed_data.size if self.num_features is None else self.num_features
        return processed_data


    def rbf_function(self, x, center, sigma):
        """
        Calculates the RBF output for a given input x and a center. 
        This function determines similarity between x and center.
        
        Parameters:
            x (ndarray): Input data point.
            center (ndarray): The center point for the RBF function.
            sigma (float): The width of the RBF function.
        
        Returns:
            float: RBF similarity value between x and the center.
        """
        return np.exp(-np.linalg.norm(x - center)**2 / (2 * sigma**2))

    def calculate_hidden_layer(self, X):
        """
        Computes the hidden layer matrix, where each entry represents the similarity 
        between a sample in X and a center in the RBF layer.
        
        Parameters:
            X (ndarray): Batch of input samples.
        
        Returns:
            ndarray: Hidden layer output with shape (number of samples, num_centers).
        """

        hidden_layer = np.zeros((X.shape[0], self.num_centers))
        for i, sample in enumerate(X):
            for j, center in enumerate(self.centers):
                hidden_layer[i, j] = self.rbf_function(sample, center, self.sigmas[j])
        return hidden_layer

    def predict(self, X):
        """
        Predicts class labels (YES/NO) for new gesture inputs by calculating
        similarity to known gestures and using the learned weights.
        
        Parameters:
            X (ndarray): Batch of new input gestures.
        
        Returns:
            ndarray: Predicted class labels (1 for YES, 0 for NO).
        """
        X_processed = np.array([self.preprocess_input(sample) for sample in X])
        hidden_layer = self.calculate_hidden_layer(X_processed)
        scores = np.dot(hidden_layer, self.weights)
        return (scores > 0).astype(int) # 将scores的元素与 0 比较，并将结果转换为整数格式（0或1）

    def train(self, X, y, lr=0.01, epochs=100):
        """
        Trains the RBF network using input gestures (X) and binary labels (y).
        Adjusts weights and centers to minimize classification error over epochs.
        
        Args:
            X (np.array): Training samples for gestures.
            y (np.array): Binary labels, where 1 indicates matching gesture, 0 for non-matching.
            lr (float): Learning rate for weight updates.
            epochs (int): Number of training iterations.
        """
        X_processed = np.array([self.preprocess_input(sample) for sample in X])
        if self.centers is None:
            self.centers = np.random.randn(self.num_centers, self.num_features)
        if self.weights is None:
            self.weights = np.random.randn(self.num_centers, self.num_classes)

        for epoch in range(epochs):
            hidden_layer = self.calculate_hidden_layer(X_processed)
            output = np.dot(hidden_layer, self.weights)
            error = y - output
            self.weights += lr * np.dot(hidden_layer.T, error)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Error: {np.mean(error ** 2)}")


# if __name__ == "__main__":
#     # 一个字典存放手势数据，用用户输入的手势名称作key
#     gesture_dict = {}
#     if inputGesture:
#         gesture_name = input("Please Enter a Gesture Name: ")
#         if gesture_name in gesture_dict.keys:
#             print(f"{gesture_name} is already exist, please use another name.")
#             gesture_name = input("Please Enter a Gesture Name: ")
#         gesture_data = record x10
#         gesture_dict[gesture_name] = Gesture(gesture_data)
#         gesture_dict[gesture_name].feature_extraction()

