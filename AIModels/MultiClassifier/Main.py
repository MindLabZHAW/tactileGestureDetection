"""
滑动窗口采集传感器数据，对数据进行特征提取，并计算与已知手势的相似度，从而判断当前的手势类别
"""
from RBFmodel import RBFNetwork
from GestureRecord import record_gesture
import numpy as np

# 初始化手势字典
gesture_dict = {}

window_length = 28
features_num = 4
dof = 7

window = np.zeros([window_length, features_num * dof])
print(f'{method}-{model.network_type}\'s window size is {window.shape}')

# 录入新手势
gesture_name = input("Please Enter a Gesture Name: ")
gesture_data =  #补充
record_gesture(gesture_name, gesture_data, gesture_dict)

# 初始化 RBF 网络
rbf_network = RBFNetwork(gesture_dict, sigma=1.0)


def contact_multiclassifier(data):
    global window 

    start_time = rospy.get_time()

    # Prepare data as done in training
    e = np.array(data.q_d) - np.array(data.q)
    # print("e is ", e)
    de = np.array(data.dq_d) - np.array(data.dq)
    tau_J = np.array(data.tau_J) 
    tau_ext = np.array(data.tau_ext_hat_filtered) 3

    new_row = np.column_stack((e,de,tau_J,tau_ext)).reshape(1, features_num * dof)
    # print(f"new row is {new_row}")
    window = np.append(window[1:, :], new_row, axis=0)
    
    # STFT
    fs = 200
    nperseg = 16
    noverlap = nperseg - 1
    data_matrix = [] 
    for feature_idx in range(window.shape[1]):
        signal = window[:, feature_idx]
        if Normalization:
            signal_mean = np.mean(signal)
            signal_std = np.std(signal)
            signal = (signal - signal_mean) / (signal_std + 1e-5)
        # f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap, window=sg.windows.general_gaussian(64, p=1, sig=7))
        f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap, window='hamming')
        data_matrix.append(np.abs(Zxx))

    data_matrix = np.concatenate(data_matrix, axis=0)  # 合并特征为输入向量

    # 计算新手势特征与已知手势的相似度
    similarity_scores = rbf_network.calculate_similarity(data_feature)

    output_dict = {}
    similarity_threshold = 90  # 设置相似度阈值 
    for gesture_name, similarity in similarity_scores.items():
        if similarity >= similarity_threshold:
            output_dict[gesture_name] = 1  # 表示可能的手势类别
        else:
            output_dict[gesture_name] = 0  # 表示不符合的手势类别

    # 输出符合条件的手势类别
    possible_gestures = [gesture for gesture, is_possible in output_dict.items() if is_possible == 1]

    if possible_gestures:
        print(f"Possible gestures: {', '.join(possible_gestures)}")
    else:
        print("No matching gesture found.")