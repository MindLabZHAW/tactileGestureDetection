"""
通过用户输入手势名称并录入手势数据，进行特征提取并存储到手势字典中

"""

import numpy as np
import os

import torch.nn as nn
import torch.optim as optim

class User(object):
    def __init__(self, user_name, password='NotSet', storage_dir='user_data'):
        self.user_name = user_name
        self.password = password

        self.gesture_dict = {}
        self.gesture_num = len(self.gesture_dict)

        self.gesture_storage_dir = os.path.join(storage_dir, self.username)
        os.makedirs(self.storage_dir, exist_ok=True)

    def get_gesture_info(self):
        gesture_list = list(self.gesture_dict.keys())
        print(f"Gesture list: {gesture_list}")
        print(f"In total {self.gesture_num} gestures")

    def add_gesture(self, gesture_name):
        if gesture_name in self.gesture_dict:
           print(f"Gesture '{gesture_name}' already exists for user '{self.user_name}'.") 
           return
        gesture = Gesture()

# 收集手势数据并提取相关特征
class Gesture(object):
    # 初始化：收集手势数据，
    def __init__(self, gesture_name, gesture_data):
        self.gesture_name = gesture_name
        self.gesture_data = gesture_data
        self.gesture_model = RBFNetwork() # 参数还没填写
        
        self.model_criterion = nn.CrossEntropyLoss()
        self.model_optimizer = optim.Adam(self.gesture_model.parameters(), lr=0.001)
        
    
    def classifier_train(self)：
        # 使用self.gesture_data访问输入数据 其实还没想好数据该在哪一步进来
        # 使用self.gesture_model实例化网络模型
    

class RBFNetwork(object):
    # 网络类直接贴过来就行

def record_gesture(gesture_name,gesture_data,gesture_dict):
    """
    记录手势数据并提取特征
    :gesture_name: 手势名称
    :gesture_data: 手势数据
    :gesture_dict: 存储手势的字典
    """
    if gesture_name in gesture_dict:
        print(f"{gesture_name} is already exist, please use another name")
    else:
        gesture = Gesture(gesture_data)
        gesture_dict[gesture_name] = gesture.gesture_feature
        print(f"Gesture {gesture_name} have been recorded")


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

