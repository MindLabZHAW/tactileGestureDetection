#!/usr/bin/env python3
"""
By Maryam Rezayati

# How to run?

#### 1st  Step: unlock robot
	-turn on the robot (wait until it has a solid yellow)
	-connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33)
	-unlock the robot
	-the robot light should be blue
	-unlock the robot and activate FCI

#### 2nd Step: run frankapy

open an terminal

	conda activate frankapyenv
	bash robotAPI/frankapy/bash_scripts/start_control_pc.sh -i localhost


#### 3rd Step: run robot node

open another terminal 

	conda activate frankapyenv
	source /opt/ros/noetic/setup.bash
	source robotAPI/franka-interface/catkin_ws/devel/setup.bash --extend
	source robotAPI/frankapy/catkin_ws/devel/setup.bash --extend
	
	$HOME/miniconda/envs/frankapyenv/bin/python3 ../tactileGestureDetection/frankaRobot/main.py

# to chage publish rate of frankastate go to : 
sudo nano /franka-interface/catkin_ws/src/franka_ros_interface/launch/franka_ros_interface.launch
"""

import os
from threading import Event
import numpy as np
import pandas as pd

import joblib
import torch
from torchvision import transforms

from collections import Counter

from scipy.signal import stft
from scipy import signal as sg
import pywt

import rospy
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from franka_interface_msgs.msg import RobotState

from frankapy import FrankaArm

from ImportModel import import_rnn_models, import_cnn_models, import_tcnn_models

# Set the main path
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
print(f"main_path is {main_path}")

# Parameters for the KNN models
window_length = 28
dof = 7
features_num = 4
classes_num = 5
method = 'TCNN'
Normalization = False


# 进行Z-score归一化-RNN
def z_score_normalization(matrix):
    # 创建一个新的矩阵以存储归一化结果
    normalized_matrix = np.empty_like(matrix)

    # 遍历每行中的每个特征（每4个元素代表一个特征）
    for row in range(matrix.shape[0]):  # 7行
        for feature_index in range(4):  # 每个特征有4个值
            # 提取该特征在28个样本中的所有值
            feature_values = matrix[row, feature_index::4]  # 提取每4个元素
            
            # 计算均值和标准差
            mean = np.mean(feature_values)
            std = np.std(feature_values)

            # 归一化
            normalized_matrix[row, feature_index::4] = (feature_values - mean) / (std + 1e-5)  # 加上一个小常数以避免除以零

    return normalized_matrix

if method == 'KNN':
    # Load the KNN model
    # model_path = '/home/weimindeqing/contactInterpretation/tactileGestureDetection/AIModels/TrainedModels/KNN_undersampling.pkl'
    model_path = '/home/mindlab/weiminDeqing/tactileGestureDetection/AIModels/TrainedModels/KNN_SVM__RF_flatten_undersampling_hybried.pkl'
    model = joblib.load(model_path)

elif method == 'RNN':
    model_path = '/home/weimindeqing/contactInterpretation/tactileGestureDetection/AIModels/TrainedModels/NCPCfC_09_26_2024_15-03-37MainPhaseKickOffMeeting.pth'
    model = import_rnn_models(model_path, network_type='NCPCfC', num_classes=classes_num, num_features=features_num, time_window=window_length)
    print(f'{method}-{model.network_type} model is loaded')

    # Set device for PyTorch models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using devise: {device}')
    if device.type == "cuda":
        torch.cuda.get_device_name()
    # Move PyTorch models to the selected device
    model = model.to(device)
    transform = transforms.Compose([transforms.ToTensor()])

elif method == 'TCNN':
    model_path = '/home/mindlab/weiminDeqing/tactileGestureDetection/AIModels/TrainedModels/2L3DTCNN_10_17_2024_19-43-34100Epoch.pth'
    model = import_tcnn_models(model_path, network_type='2L3DTCNN', num_classes=classes_num, num_features=features_num, time_window=window_length)
    print(f'{method}-{model.network_type} model is loaded')

    # Set device for PyTorch models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using devise: {device}')
    if device.type == "cuda":
        torch.cuda.get_device_name()
    # Move PyTorch models to the selected device
    model = model.to(device)
    # transform = transforms.Compose([transforms.ToTensor()]) # ToTensor will automatically change (H,W,C) to (C, H, W) so abort

elif method == 'Freq':
    model_path = '/home/weimindeqing/contactInterpretation/tactileGestureDetection/AIModels/TrainedModels/T2L3DCNN_10_17_2024_13-40-57normalization.pth'
    model = import_cnn_models(model_path, network_type='2L3DCNN', num_classes=classes_num)
    print(f'{method}-{model.network_type} model is loaded')

    # Set device for PyTorch models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using devise: {device}')
    if device.type == "cuda":
        torch.cuda.get_device_name()
    # Move PyTorch models to the selected device
    model = model.to(device)
    transform = transforms.Compose([transforms.ToTensor()]) # Raw input is (H,W,C) so use ToTensor() to transform to (C,H,W)



# Prepare window to collect features
if method == 'KNN':
    window = np.zeros([1, window_length * features_num * dof])
    print(f'{method}-{model.network_type}\'s window size is {window.shape}')
elif method == 'RNN':
    window = np.zeros([dof, features_num * window_length])
    window2 = np.zeros([dof, features_num * window_length])
    window3 = np.zeros([dof, features_num * window_length])
    print(f'{method}-{model.network_type}\'s window size is {window.shape}')
elif method == 'TCNN':
    window = np.zeros([window_length, dof, features_num])
    print(f'{method}-{model.network_type}\'s window size is {window.shape}')
elif method == 'Freq':
    window = np.zeros([window_length, features_num * dof])
    print(f'{method}-{model.network_type}\'s window size is {window.shape}')


# Initialize a list to store the results
results = []
# Create message for publishing model output (will be used in saceDataNode.py)
model_msg = Floats()


# Callback function for contact detection and prediction
def contact_detection(data):
    global window, window2, window3, results, big_time_digits

    start_time = rospy.get_time()

    # Prepare data as done in training
    e = np.array(data.q_d) - np.array(data.q)
    # print("e is ", e)
    de = np.array(data.dq_d) - np.array(data.dq)
    tau_J = np.array(data.tau_J)  
    tau_ext = np.array(data.tau_ext_hat_filtered)

    if Normalization:
        mean_e, std_e = np.mean(e), np.std(e)
        mean_de, std_de = np.mean(de), np.std(de)
        mean_tau_J, std_tau_J = np.mean(tau_J), np.std(tau_J)
        mean_tau_ext, std_tau_ext = np.mean(tau_ext), np.std(tau_ext)


        e = (e - mean_e) / (std_e+ 1e-5)
        de = (de - mean_de) / (std_de + 1e-5)
        tau_J = (tau_J - mean_tau_J) / (std_tau_J+ 1e-5)
        tau_ext = (tau_ext - mean_tau_ext) / (std_tau_ext + 1e-5)


    
    if method == 'KNN':
        new_data = np.column_stack((e,de,tau_J,tau_ext)).reshape(1, -1)
        # print(f"new data is {new_data}")
        # print(f"new data size is{new_data.shape}")
        
        window = np.append(window[:,features_num * dof:], new_data, axis=1)

        # Predict the touch_type using the KNN model
        touch_type_idx = model.predict(window)[0]
        touch_type = label_map_inv[touch_type_idx]  

        # Store the results
        time_sec = int(rospy.get_time())
        results.append([time_sec, touch_type])

    elif method == 'RNN':
        new_block = np.column_stack((e,de,tau_J,tau_ext))
        print(f"new block is {new_block}")
        normalized_new_block = z_score_normalization(new_block)
        print(f"normalized_new_block is {normalized_new_block}")
        # print(f"front block is {window[:, features_num:].shape}")
        window3 = window2
        window2 = window
        window = np.append(window[:, features_num:], new_block, axis=1)

        with torch.no_grad():
            # Prepare inputs
            data_input1 = transform(window).to(device).float()
            data_input2 = transform(window2).to(device).float()
            data_input3 = transform(window3).to(device).float()
            # Calculate model outputs for each window
            model_out1 = model(data_input1).detach()
            model_out2 = model(data_input2).detach()
            model_out3 = model(data_input3).detach()
            # Get predictions for each window 
            output1 = torch.argmax(model_out1, dim=1)
            output2 = torch.argmax(model_out2, dim=1)
            output3 = torch.argmax(model_out3, dim=1)
        # Convert outputs to CPU numpy arrays
        output1_idx = output1.cpu().numpy()[0]
        output2_idx = output2.cpu().numpy()[0]
        output3_idx = output3.cpu().numpy()[0]
        # Collect outputs
        outputs_idx = [output1_idx, output2_idx, output3_idx]
        # Perform majority voting
        touch_type_idx = Counter(outputs_idx).most_common(1)[0][0]
        # Project the int index to string output
        label_map_RNN = {0:"NC", 1:"ST", 2:"DT", 3:"P", 4:"G"}
        touch_type = label_map_RNN[touch_type_idx]  # Get the actual touch type label
        # Store the results
        time_sec = int(rospy.get_time())
        results.append([time_sec, touch_type])
    

    elif method == 'TCNN':
        new_block = np.expand_dims(np.column_stack((e,de,tau_J,tau_ext)), axis=0)
        # print(new_block.shape)
        window = np.append(window[1:, :, :], new_block, axis = 0)
        with torch.no_grad():
            # Prepare inputs
            # print(window.shape)
            data_input = torch.from_numpy(window).unsqueeze(0).to(device).float()
            # Calculate model outputs for each window
            # print(data_input.shape)
            model_out = model(data_input).detach()
            # Get predictions for each window 
            output = torch.argmax(model_out, dim=1)
        # Convert outputs to CPU numpy arrays
        touch_type_idx = output.cpu().numpy()[0]
        # Project the int index to string output
        label_map_RNN = {0:"NC", 1:"ST", 2:"DT", 3:"P", 4:"G"}
        touch_type = label_map_RNN[touch_type_idx]  # Get the actual touch type label
        # Store the results
        time_sec = int(rospy.get_time())
        results.append([time_sec, touch_type])


    elif method == 'Freq':
        new_row = np.column_stack((e,de,tau_J,tau_ext)).reshape(1, features_num * dof)
        # print(f"new row is {new_row}")
        window = np.append(window[1:, :], new_row, axis=0)
        
        if model.network_type in ['2LCNN', '2L3DCNN']:
            # STFT
            fs = 200
            nperseg = 16
            noverlap = nperseg - 1
            data_matrix = [] 
            for feature_idx in range(window.shape[1]):
                # f, t, Zxx = stft([:, feature_idx], fs, nperseg=nperseg, noverlap=noverlap, window=sg.windows.general_gaussian(64, p=1, sig=7))
                f, t, Zxx = stft(window[:, feature_idx], fs, nperseg=nperseg, noverlap=noverlap, window='hamming')
                data_matrix.append(np.abs(Zxx))
        elif model.network_type == '3LCNN':
            # CWT
            wavelet = 'cmor'  # Complex Morlet wavelet (suitable for time-frequency analysis)
            scales = np.arange(1, 128)  # Adjust the range of scales as needed
            fs = 200
            data_matrix = [] 
            for feature_idx in range(window.shape[1]):
                coefficients, frequencies = pywt.cwt(window[:, feature_idx], scales, wavelet,sampling_period=1/fs)
                data_matrix.append(np.abs(coefficients))
        elif model.network_type == 'T2L3DCNN':
            T_window_size = 16
            T_step = 1  # not used because iteration automatically +1,but if step is not 1 we need to fix this
            T_window_num = len(window) - T_window_size + 1
            data_matrix = [] 
            for feature_idx in range(window.shape[1]):
                T_matrix = np.zeros([T_window_size, T_window_num])
                signal = window[:, feature_idx]
                for i in range(T_window_num):
                    T_matrix[:,i] = signal[i:i+T_window_size]
                # print(T_matrix)
                data_matrix.append(np.array(T_matrix))

        # Predict the touch_type using the CNN Model
        with torch.no_grad():
            stft_matrix = np.stack(data_matrix, axis=-1)
            # print(stft_matrix.shape)
            # print(torch.unsqueeze(transform(stft_matrix),0).shape)
            stft_matrix_input = transform(stft_matrix).unsqueeze(0).to(device).float()
            # print(stft_matrix_input.shape)
            model_out = model(stft_matrix_input).detach()
            # print(model_out)
            output = torch.argmax(model_out)
            # print(output)
        touch_type_idx = int(output.cpu())
        label_map_RNN = { 0:"NC",1:"ST", 2:"DT", 3:"P", 4:"G"}
        touch_type = label_map_RNN[touch_type_idx]  # Get the actual touch type label

        # Store the results
        time_sec = int(rospy.get_time())
        results.append([time_sec, touch_type])



    # Log prediction
    detection_duration  = rospy.get_time() - start_time
    rospy.loginfo(f'Predicted touch_type: {touch_type} and the detection duration is {detection_duration}')
    
    start_time = np.array(start_time).tolist()
    time_sec = int(start_time)
    time_nsec = start_time-time_sec
    model_msg.data = np.append(np.array([time_sec-big_time_digits, time_nsec, detection_duration, touch_type_idx], dtype=np.complex128), np.hstack(window))
    # print(model_msg)
    model_pub.publish(model_msg)

if __name__ == "__main__":
    global publish_output, big_time_digits

    # Load inverse label map for decoding predictions
    label_map_inv = {0:"NC",1:"ST", 2:"DT", 3:"P", 4:"G"}
    label_classes_RNN = { 0:"NC",1:"ST", 2:"DT", 3:"P", 4:"G"}
    event = Event()
    
    # Create robot controller instance
    fa = FrankaArm()

    scale = 1000000
    big_time_digits = int(rospy.get_time()/scale)*scale

    # Subscribe to robot data topic for contact detection module
    rospy.Subscriber(name="/robot_state_publisher_node_1/robot_state", data_class=RobotState, callback=contact_detection)
    model_pub = rospy.Publisher("/model_output", numpy_msg(Floats), queue_size= 1)

    # Run the ROS event loop
    rospy.spin()

    """ # After ROS loop ends, save results to a CSV file
    results_df = pd.DataFrame(results, columns=['time_sec', 'predicted_touch_type'])
    results_path = main_path + 'predicted_touch_types.csv'
    results_df.to_csv(results_path, index=False)
    print(f'Predicted touch types saved to {results_path}')
 """