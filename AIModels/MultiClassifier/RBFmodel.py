import numpy as np
from sklearn.cluster import KMeans

class RBFNetwork:
    def __init__(self,gestures_features_dict,sigma=1.0):
        """
        初始化RBF网络，用于计算新手势与已知多个手势类别的相识度
        ：gestures_features_dict:字典，每个键是手势类别名称，值是手势类别的特征矩阵
        ：sigma:RBF 函数的宽度参数
        """
        self.gestures_features_dict = gestures_features_dict
        self.sigma = sigma

    def rbf(self,x,centers):
        """
        计算输入数据与中心的RBF值
        ：x:输入数据
        ：center:RBF中心
        ：return:RBF输出s
        """
        rbf_value = np.exp(-np.linalg.norm(x[:, None] - centers, axis=2)**2 / (2 * self.sigma**2))
        return rbf_value

    def calculate_similarity(self,new_gesture_feature):
        """
        计算新手势与所有已知手势类别的相似度
        ：new_gesture_feature:新手势的特征向量
        ：return: 字典，与每个手势类别的相识度
        """
        similarity_scores = {}
        for gesture_name,feature in self.gesture_features_dict.items():
            similarities = self.rbf(new_gesture_feature.reshape(1, -1), features).flatten()
            similarity_scores[gesture_name] = np.mean(similarities) # 计算每个手势类别相似度均值
        return similarity_scores

    