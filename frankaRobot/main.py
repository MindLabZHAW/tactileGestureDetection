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
	
	$HOME/miniconda/envs/frankapyenv/bin/python3 tactileGestureDetection/frankaRobot/main.py

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
from scipy.signal import stft
from scipy import signal as sg

import rospy
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from franka_interface_msgs.msg import RobotState

from frankapy import FrankaArm

from ImportModel import import_rnn_models

# Set the main path
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
print(f"main_path is {main_path}")

# Parameters for the KNN models
window_length = 200
dof = 7
features_num = 4
classes_num = 5
method = 'RNN'

if method == 'KNN':
    # Load the KNN model
    model_path = '/home/weimindeqing/contactInterpretation/tactileGestureDetection/AIModels/TrainedModels/trained_knn_model.pkl'
    model = joblib.load(model_path)
elif method == 'RNN':
    model_path = '/home/weimindeqing/contactInterpretation/tactileGestureDetection/AIModels/TrainedModels/LSTM_08_27_2024_19-46-06.pth'
    model = import_rnn_models(model_path, network_type='LSTM', num_classes=classes_num, num_features=features_num, time_window=window_length)

    # Set device for PyTorch models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    if device.type == "cuda":
        torch.cuda.get_device_name()
    # Move PyTorch models to the selected device
    model = model.to(device)
    transform = transforms.Compose([transforms.ToTensor()])
elif method == 'Freq':
    model_path = 'AIModels/TrainedModels/trained_knn_model.pkl'
    # model = 


# Prepare window to collect features
if method == 'KNN' or method ==' Freq':
    window = np.zeros([window_length, features_num * dof])
elif method == 'RNN':
    window = np.zeros([dof, features_num * window_length])

# Initialize a list to store the results
results = []

# Callback function for contact detection and prediction
def contact_detection(data):
    global window, results

    # Prepare data as done in training
    e_q = np.array(data.q_d) - np.array(data.q)
    e_dq = np.array(data.dq_d) - np.array(data.dq)
    tau_J = np.array(data.tau_J)  
    tau_ext = np.array(data.tau_ext_hat_filtered)

    # q = np.array(data.q)
    # q_d = np.array(data.q_d)
    # dq = np.array(dq)
    # dq_d = np.array(dq_d)
    # tau_J_d = np.array(tau_J_d)
    # e = 
    # de 

    if method == 'KNN':
        # Create new row and update the sliding window
        new_row = np.hstack((tau_J,tau_ext,e_q, e_dq)).reshape((1, features_num * dof))
        # print(f"new row is {new_row}")
        window = np.append(window[1:, :], new_row, axis=0)

        # Flatten the window to create a feature vector for the model
        feature_vector = window.mean(axis=0).reshape(1, -1)

        # Predict the touch_type using the KNN model
        touch_type_idx = model.predict(feature_vector)[0]
        touch_type = label_map_inv[touch_type_idx]  # Get the actual touch type label

        # Store the results
        time_sec = int(rospy.get_time())
        results.append([time_sec, touch_type])


    elif method == 'RNN':
        new_block = np.column_stack((e_q,e_dq,tau_J,tau_ext))
        # print(f"new block is {new_block}")
        # print(f"front block is {window[:, features_num:].shape}")
        window = np.append(window[:, features_num:], new_block, axis=1)

        with torch.no_grad():
            data_input = transform(window).to(device).float()
            model_out = model(data_input)
            model_out = model_out.detach()
            output = torch.argmax(model_out, dim=1)
        touch_type_idx = output.cpu().numpy()[0]
        label_map_RNN = {0:"ST", 1:"DT", 2:"P", 3:"G", 4:"NC"}
        touch_type = label_map_RNN[touch_type_idx]  # Get the actual touch type label
        # Store the results
        time_sec = int(rospy.get_time())
        results.append([time_sec, touch_type])


    elif method == 'Freq':
        new_row = np.column_stack((e_q,e_dq,tau_J,tau_ext)).reshape(1, features_num * dof)
        print(f"new row is {new_row}")
        window = np.append(window[1:, :], new_row, axis=0)
        
        # STFT 
        fs = 100
        nperseg = 64
        noverlap = nperseg // 2
        f, t, Zxx = stft(window, fs, nperseg=nperseg, noverlap=noverlap, window=sg.windows.general_gaussian(64, p=1, sig=7))
        feature_vector = abs(Zxx)

        # Predict the touch_type using the KNN model
        touch_type_idx = model.predict(feature_vector)[0]
        touch_type = label_map_inv[touch_type_idx]  # Get the actual touch type label

        # Store the results
        time_sec = int(rospy.get_time())
        results.append([time_sec, touch_type])



    # Log prediction
    rospy.loginfo(f'Predicted touch_type: {touch_type}')

if __name__ == "__main__":
    global publish_output, big_time_digits

    # Load inverse label map for decoding predictions
    label_classes = ['DT','G','P','ST']  # Update according to your dataset
    label_map_inv = {idx: label for idx, label in enumerate(label_classes)}
    label_classes_RNN = {0:"ST", 1:"DT", 2:"P", 3:"G", 4:"NC"}
    event = Event()
    
    # Create robot controller instance
    fa = FrankaArm()

    # Subscribe to robot data topic for contact detection module
    rospy.Subscriber(name="/robot_state_publisher_node_1/robot_state", data_class=RobotState, callback=contact_detection)

    # Run the ROS event loop
    rospy.spin()

    """ # After ROS loop ends, save results to a CSV file
    results_df = pd.DataFrame(results, columns=['time_sec', 'predicted_touch_type'])
    results_path = main_path + 'predicted_touch_types.csv'
    results_df.to_csv(results_path, index=False)
    print(f'Predicted touch types saved to {results_path}')
 """