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

from ImportModel import import_rnn_models, import_cnn_models

# Set the main path
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
print(f"main_path is {main_path}")

# Parameters for the KNN models
window_length = 28
dof = 7
features_num = 4
classes_num = 5
method = 'RNN'

if method == 'KNN':
    # Load the KNN model
    model_path = '/home/weimindeqing/contactInterpretation/tactileGestureDetection/AIModels/TrainedModels/KNN.pkl'
    model = joblib.load(model_path)
elif method == 'RNN':
    model_path = '/home/weimindeqing/contactInterpretation/tactileGestureDetection/AIModels/TrainedModels/NCPCfC_09_12_2024_15-27-52NewData.pth'
    model = import_rnn_models(model_path, network_type='NCPCfC', num_classes=classes_num, num_features=features_num, time_window=window_length)

    # Set device for PyTorch models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    if device.type == "cuda":
        torch.cuda.get_device_name()
    # Move PyTorch models to the selected device
    model = model.to(device)
    transform = transforms.Compose([transforms.ToTensor()])
elif method == 'Freq':
    model_path = '/home/weimindeqing/contactInterpretation/tactileGestureDetection/AIModels/TrainedModels/2LCNN_09_19_2024_17-48-27.pth'
    model = import_cnn_models(model_path, network_type='2LCNN', num_classes=classes_num)

    # Set device for PyTorch models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    if device.type == "cuda":
        torch.cuda.get_device_name()
    # Move PyTorch models to the selected device
    model = model.to(device)
    transform = transforms.Compose([transforms.ToTensor()])


# Prepare window to collect features
if method == 'KNN':
    window = np.zeros([1, window_length * features_num * dof])
elif method == 'RNN':
    window = np.zeros([dof, features_num * window_length])
elif method == 'Freq':
    window = np.zeros([window_length, features_num * dof])


# Initialize a list to store the results
results = []
# Create message for publishing model output (will be used in saceDataNode.py)
model_msg = Floats()

# Callback function for contact detection and prediction
def contact_detection(data):
    global window, results, big_time_digits

    start_time = rospy.get_time()

    # Prepare data as done in training
    e = np.array(data.q_d) - np.array(data.q)
    # print("e is ", e)
    de = np.array(data.dq_d) - np.array(data.dq)
    tau_J = np.array(data.tau_J)  
    tau_ext = np.array(data.tau_ext_hat_filtered)

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
        # print(f"new block is {new_block}")
        # print(f"front block is {window[:, features_num:].shape}")
        window = np.append(window[:, features_num:], new_block, axis=1)

        with torch.no_grad():
            data_input = transform(window).to(device).float()
            model_out = model(data_input)
            model_out = model_out.detach()
            # print(model_out)
            output = torch.argmax(model_out, dim=1)
            # print(output)
        touch_type_idx = output.cpu().numpy()[0]
        label_map_RNN = {0:"ST", 1:"DT", 2:"P", 3:"G", 4:"NC"}
        touch_type = label_map_RNN[touch_type_idx]  # Get the actual touch type label
        # Store the results
        time_sec = int(rospy.get_time())
        results.append([time_sec, touch_type])


    elif method == 'Freq':
        new_row = np.column_stack((e,de,tau_J,tau_ext)).reshape(1, features_num * dof)
        # print(f"new row is {new_row}")
        window = np.append(window[1:, :], new_row, axis=0)
        
        # STFT
        fs = 200
        nperseg = 16
        noverlap = nperseg - 1
        data_matrix = [] 
        for feature_idx in range(window.shape[1]):
            # f, t, Zxx = stft([:, feature_idx], fs, nperseg=nperseg, noverlap=noverlap, window=sg.windows.general_gaussian(64, p=1, sig=7))
            f, t, Zxx = stft(window[:, feature_idx], fs, nperseg=nperseg, noverlap=noverlap, window='hamming')
            data_matrix.append(np.abs(Zxx))

        # Predict the touch_type using the CNN Model
        with torch.no_grad():
            stft_matrix = np.stack(data_matrix, axis=-1)
            stft_matrix_input = transform(stft_matrix).to(device).float()
            model_out = model(stft_matrix_input)
            model_out = model_out.detach()
            # print(model_out)
            output = torch.argmax(model_out)
            # print(output)
        touch_type_idx = int(output.cpu())
        label_map_RNN = {0:"ST", 1:"DT", 2:"P", 3:"G", 4:"NC"}
        touch_type = label_map_RNN[touch_type_idx]  # Get the actual touch type label

        # Store the results
        time_sec = int(rospy.get_time())
        results.append([time_sec, touch_type])



    # Log prediction
    detection_duration  = rospy.get_time() - start_time
    rospy.loginfo(f'Predicted touch_type: {touch_type}')
    
    start_time = np.array(start_time).tolist()
    time_sec = int(start_time)
    time_nsec = start_time-time_sec
    model_msg.data = np.append(np.array([time_sec-big_time_digits, time_nsec, detection_duration, touch_type_idx], dtype=np.complex128), np.hstack(window))
    # print(model_msg)
    model_pub.publish(model_msg)

if __name__ == "__main__":
    global publish_output, big_time_digits

    # Load inverse label map for decoding predictions
    label_classes = ['DT','G','NC','P','ST'] 
    label_map_inv = {idx: label for idx, label in enumerate(label_classes)}
    label_classes_RNN = {0:"ST", 1:"DT", 2:"P", 3:"G", 4:"NC"}
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