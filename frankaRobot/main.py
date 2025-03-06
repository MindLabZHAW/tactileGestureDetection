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
    In this line, you may have to change the script path according to your environment

# to chage publish rate of frankastate go to : 
sudo nano /franka-interface/catkin_ws/src/franka_ros_interface/launch/franka_ros_interface.launch
"""
# Import general libraries
import os
import pickle
from threading import Event
import numpy as np
import pandas as pd
from collections import Counter

# Import method related libraries
# General
import torch
from torchvision import transforms
# KNN model
import joblib 
# Frequency model
from scipy.signal import stft
from scipy import signal as sg
import pywt

# Import rospy and Frankapy related libraries
# rospy
import rospy
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from franka_interface_msgs.msg import RobotState
# Frankapy
from frankapy import FrankaArm

# Set the main path
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
print(f"main_path is {main_path}")

# Import our own script
# ImportModel Used to import different models' structure
from ImportModel import import_rnn_models, import_cnn_models, import_tcnn_models
# Import Multiple Classifiers form "/AIModels/MultiClassifier"
import sys
sys.path.append(os.path.join(main_path,"AIModels","MultiClassifier"))
from GestureRecord_softmax import Gesture, RBFNetwork

# General Hyper-Parameters
window_length = 28
dof = 7
features_num = 4
classes_num = 4
Normalization = False
# Model Selection Parameters and path (Please always change them together)
method = 'Freq'
model_path_relative = os.path.join("AIModels/TrainedModels", "STFT3DCNN_03_05_2025_17-17-15Pose123ES60Conv4+7_STFTBound.pth") # relative path from AIModels
type_network = 'STFT3DCNN'
# MultiClassifier Parameters and path
MultiClassifier = False
if MultiClassifier:
    user_folder_path = os.path.join(main_path, "user_data/DS/gesture_pickle")
# Majority Vote or not?
MajorityVote = True



# Z-score Normalization Function for RNN
def z_score_normalization(matrix):
    # Create an empty matrix to store the Normalized data
    normalized_matrix = np.empty_like(matrix) # 7 rows by 4*window_length columns

    # Iterate over every feature in every joint
    for row in range(matrix.shape[0]):  # 7 joints in 7 rows
        for feature_index in range(features_num):  # 4 features

            feature_values = matrix[row, feature_index::4]  # extract every 4 columns
            
            # Calculate mean and std for each feature-joint combinition
            mean = np.mean(feature_values)
            std = np.std(feature_values)

            # Z-score Normalization
            normalized_matrix[row, feature_index::4] = (feature_values - mean) / (std + 1e-5)  # Add a small number to avoid divided by 0 

    return normalized_matrix

# Loading Models
if method == 'KNN':
    # Load KNN model
    model_path = os.path.join(main_path, model_path_relative)
    model = joblib.load(model_path)

elif method == 'RNN':
    # Load RNN Model
    '''
    Possible network_type: LSTM, GRU, FCLTC, FCCfC, NCPLTC, NCP, CfC
    Notation: 
        FC for Fully-Connected Network stucture, NCP for Neural Circuit Policy Nw sturcture
        LTC for Liquid Time-Constant Neuron, CfC for Closed-Form Continuous-Time Neuron 
    '''
    model_path = os.path.join(main_path, model_path_relative)
    model = import_rnn_models(model_path, network_type=type_network, num_classes=classes_num, num_features=features_num, time_window=window_length)
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
    # Load Time CNN model
    '''
    Possible network_type: 1L3DTCNN, 2L3DTCNN
    Use 3D Image (Time as channel axis) combining with 3D CNN
    '''
    model_path = os.path.join(main_path, model_path_relative)
    model = import_tcnn_models(model_path, network_type=type_network, num_classes=classes_num, num_features=features_num, time_window=window_length)
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
    # Load Freqency Model
    '''
    Possible network_type: 2LCNN, 3LCNN, 2L3DCNN, T2L3DCNN, STFT3DCNN, STT3DCNN
    Use 2D or 3D(Features as channel axis) Image with 2D/3D CNN 
    T2L3DCNN is a fake Spectrogram with no frequency transform but raw time domain data in each channel(feature) 
    '''
    model_path = os.path.join(main_path, model_path_relative)
    model = import_cnn_models(model_path, network_type= type_network, num_classes=classes_num)
    print(f'{method}-{model.network_type} model is loaded')

    # Set device for PyTorch models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using devise: {device}')
    if device.type == "cuda":
        torch.cuda.get_device_name()
    # Move PyTorch models to the selected device
    model = model.to(device)
    transform = transforms.Compose([transforms.ToTensor()]) # Raw input is (H,W,C) so use ToTensor() to transform to (C,H,W)

# Load Multi Classifier Models
if MultiClassifier:
    user_file_list = os.listdir(user_folder_path)
    gesture_list = []
    # Load All Gesture Object
    print(user_file_list)
    for gesture_file in user_file_list:
        print(gesture_file)
        with open(os.path.join(user_folder_path, gesture_file), 'rb') as file:
            Gesture_load = pickle.load(file)
            gesture_list.append(Gesture_load)


# Prepare window to collect features
if method == 'KNN':
    window = np.zeros([1, window_length * features_num * dof])
    print(f'{method}-{type_network}\'s window size is {window.shape}')
elif method == 'RNN':
    window = np.zeros([dof, features_num * window_length])
    window2 = np.zeros([dof, features_num * window_length])
    window3 = np.zeros([dof, features_num * window_length])
    print(f'{method}-{model.network_type}\'s window size is {window.shape}')
elif method == 'TCNN':
    window = np.zeros([window_length, dof, features_num])
    window2 = np.zeros([window_length, dof, features_num])
    window3 = np.zeros([window_length, dof, features_num])
    print(f'{method}-{model.network_type}\'s window size is {window.shape}')
elif method == 'Freq':
    window = np.zeros([window_length, features_num * dof])
    if MajorityVote:
        stft_matrix = np.zeros([9, 29, features_num * dof])
        stft_matrix2 = np.zeros([9, 29, features_num * dof])
        stft_matrix3 = np.zeros([9, 29, features_num * dof])
        output1_idx = 0
        output2_idx = 0
        output3_idx = 0
    print(f'{method}-{model.network_type}\'s window size is {window.shape}')


# Initialize a list to store the results
results = []
# Create message for publishing model output (will be used in saveDataNode.py)
model_msg = Floats()


# Callback function for contact detection and gesture prediction
def contact_detection(data):
    global window, window2, window3, results, big_time_digits
    if MajorityVote:
        # global stft_matrix, stft_matrix2, stft_matrix3
        global output1_idx, output2_idx, output3_idx

    start_time = rospy.get_time()

    # Extract needed features from data
    e = np.array(data.q_d) - np.array(data.q) # 7D vector
    # print("e is ", e)
    de = np.array(data.dq_d) - np.array(data.dq) # 7D vector
    tau_J = np.array(data.tau_J) # 7D vector
    tau_ext = np.array(data.tau_ext_hat_filtered) # 7D vector


    # KNN Procedure
    if method == 'KNN':
        # Reshape the new comming data
        new_data = np.column_stack((e,de,tau_J,tau_ext)).reshape(1, -1)
        # print(f"new data is {new_data}")
        # print(f"new data size is{new_data.shape}")
        
        # Shift the window to contain the newest data and exclude the oldest data
        window = np.append(window[:,features_num * dof:], new_data, axis=1)

        # Predict the touch_type using the KNN model
        touch_type_idx = model.predict(window)[0]
        touch_type = label_map_inv[touch_type_idx]  

        # Store the results
        time_sec = int(rospy.get_time())
        results.append([time_sec, touch_type])
    
    # RNN Procedure
    elif method == 'RNN':
        # Reshape the new comming data
        new_block = np.column_stack((e,de,tau_J,tau_ext))
        # print(f"new block is {new_block}")
        # print(f"front block is {window[:, features_num:].shape}")
        # Update window3(t-2) and window2(t-1) and window(t) in sequence
        window3 = window2
        window2 = window
        window = np.append(window[:, features_num:], new_block, axis=1)
        # Normalization
        if Normalization:
            window = z_score_normalization(window)
        # Use the network to predict in all 3 windows
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
        if classes_num == 5:
            label_map_RNN = { 0:"NC",1:"ST", 2:"DT", 3:"P", 4:"G"}
        elif classes_num == 4:
            label_map_RNN = { 0:"NC",1:"ST", 2:"P", 3:"G"}
        else:   
            print("Wrong Class Number in Label Map RNN")
        touch_type = label_map_RNN[touch_type_idx]  # Get the actual touch type label
        # Store the results
        time_sec = int(rospy.get_time())
        results.append([time_sec, touch_type])
    
    # TCNN Procedure
    elif method == 'TCNN':
        # Reshape the new comming data
        new_block = np.expand_dims(np.column_stack((e,de,tau_J,tau_ext)), axis=0)
        # print(new_block.shape)
        
        # Update window3(t-2) and window2(t-1) and window(t) in sequence
        window3 = window2
        window2 = window
        # Shift the window to contain the newest data and exclude the oldest data
        window = np.append(window[1:, :, :], new_block, axis = 0)
        
        # Normalization
        if Normalization:
            window_mean = np.mean(window, axis=0)
            window_std = np.std(window, axis=0)
            window = (window - window_mean) / (window_std + 1e-5)
        
        # Predict the touch_type using the CNN model with T-Image
        # Use the network to predict in all 3 windows
        with torch.no_grad():
            # Prepare inputs
            # print(window.shape)
            data_input1 = torch.from_numpy(window).unsqueeze(0).to(device).float()
            data_input2 = torch.from_numpy(window2).unsqueeze(0).to(device).float()
            data_input3 = torch.from_numpy(window3).unsqueeze(0).to(device).float()
            
            # Calculate model outputs for each window
            model_out1 = model(data_input1).detach()
            model_out2 = model(data_input2).detach()
            model_out3 = model(data_input3).detach()
            # print(data_input.shape)

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
        if classes_num == 5:
            label_map_TCNN = { 0:"NC",1:"ST", 2:"DT", 3:"P", 4:"G"}
        elif classes_num == 4:
            label_map_TCNN = { 0:"NC",1:"ST", 2:"P", 3:"G"}
        else:   
            print("Wrong Class Number in Label Map TCNN")
        touch_type = label_map_TCNN[touch_type_idx]  # Get the actual touch type label
        # Store the results
        time_sec = int(rospy.get_time())
        results.append([time_sec, touch_type])


    elif method == 'Freq':
        # Reshape the new comming data
        new_row = np.column_stack((e,de,tau_J,tau_ext)).reshape(1, features_num * dof)
        # print(f"new row is {new_row}")
        # Shift the window to contain the newest data and exclude the oldest data
        window = np.append(window[1:, :], new_row, axis=0)
       
        # 2 Layers 2D/3D CNN - STFT
        if model.network_type in ['2LCNN', '2L3DCNN', 'STFT3DCNN']:
            # Hyperparameters for STFT
            fs = 200 # sample frequency
            nperseg = 16 # STFT window size
            noverlap = nperseg - 1 # STFT window stride
            
            data_matrix = []
            # Iterate over every feature in every joint
            for feature_idx in range(window.shape[1]):
                # Extract signal for each feature in every joint
                signal = window[:, feature_idx] 
                # Normalization on single signal
                if Normalization:
                    signal_mean = np.mean(signal)
                    signal_std = np.std(signal)
                    signal = (signal - signal_mean) / (signal_std + 1e-5)
                # Conduct Short-time Fourier Transform
                # f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap, window=sg.windows.general_gaussian(64, p=1, sig=7))
                # f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap, window='hamming')
                f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap, window='hamming', boundary=None)
                # Stack signals in 3rd dimension
                data_matrix.append(np.abs(Zxx))
        # 3 Layers 2D CNN - CWT
        elif model.network_type == '3LCNN':
            # Hyperparameters for CWT
            wavelet = 'cmor'  # Complex Morlet wavelet (suitable for time-frequency analysis)
            scales = np.arange(1, 128)  # Adjust the range of scales as needed
            fs = 200 # sample frequency

            data_matrix = [] 
            # Iterate over every feature in every joint
            for feature_idx in range(window.shape[1]):
                # Extract signal for each feature in every joint
                signal = window[:, feature_idx]  
                # Conduct Continuous Wavelet Transform
                coefficients, frequencies = pywt.cwt(signal, scales, wavelet,sampling_period=1/fs)
                # Stack signals in 3rd dimension
                data_matrix.append(np.abs(coefficients))
        # 2 Layer 3D CNN - Fake STFT Image on Time Domain
        elif model.network_type in ['T2L3DCNN', 'STT3DCNN']:
            # Hyperparameters
            T_window_size = 16 # window size
            T_step = 1  # not used because iteration automatically +1,but if step is not 1 we need to fix this
            T_window_num = len(window) - T_window_size + 1 # calculate the total window number
            
            data_matrix = [] 
            # Iterate over every feature in every joint
            for feature_idx in range(window.shape[1]):
                # Using directly each signal on time domain instead of conduction frequency domain transform 
                T_matrix = np.zeros([T_window_size, T_window_num])
                # Extract signal for each feature in every joint
                signal = window[:, feature_idx]
                # Normalization
                if Normalization:
                    signal_mean = np.mean(signal)
                    signal_std = np.std(signal)
                    signal = (signal - signal_mean) / (signal_std + 1e-5)
                # Put each window directly into columns(No transform)
                for i in range(T_window_num):
                    T_matrix[:,i] = signal[i:i+T_window_size]
                # Stack signals in 3rd dimension
                # print(T_matrix)
                data_matrix.append(np.array(T_matrix))

        # Predict the touch_type using the CNN Model with F-Image
        with torch.no_grad():
            # Change the multiple 2D data_matrix in list into stacked 3D stft_matrix
            # if MajorityVote:
            #     stft_matrix3 = stft_matrix2
            #     stft_matrix2 = stft_matrix
            stft_matrix = np.stack(data_matrix, axis=-1)
            # print(stft_matrix.shape)
            # print(torch.unsqueeze(transform(stft_matrix),0).shape)
            # Prepare inputs
            # transform 3rd dimension as channel dimension [(F,T,C) to (C,F,T)]
            stft_matrix_input = transform(stft_matrix).unsqueeze(0).to(device).float()
            # if MajorityVote:
            #     stft_matrix_input2 = transform(stft_matrix2).unsqueeze(0).to(device).float()
            #     stft_matrix_input3 = transform(stft_matrix3).unsqueeze(0).to(device).float()
            
            # print(stft_matrix_input.shape)
            # Calculate model outputs
            model_out1 = model(stft_matrix_input).detach()
            # if MajorityVote:
            #     model_out2 = model(stft_matrix_input2).detach()
            #     model_out3 = model(stft_matrix_input3).detach()
            # print(model_out)
            # Get predictions
            output1_idx = int(torch.argmax(model_out1).cpu())
            # if MajorityVote:
            #     output2_idx = int(torch.argmax(model_out2).cpu())
            #     output3_idx = int(torch.argmax(model_out3).cpu())
            # Collect outputs
            if MajorityVote:
                output3_idx = output2_idx
                output2_idx = output1_idx
                outputs_idx = [output1_idx, output2_idx, output3_idx]
                # Perform majority voting
                touch_type_idx = Counter(outputs_idx).most_common(1)[0][0]
            else:
                touch_type_idx = output1_idx
            # print(output)
        # Project the int index to string output
        if classes_num == 5:
            label_map_Freq = { 0:"NC",1:"ST", 2:"DT", 3:"P", 4:"G"}
        elif classes_num == 4:
            label_map_Freq = { 0:"NC",1:"ST", 2:"P", 3:"G"}
        else:   
            print("Wrong Class Number in Label Map Freq")
        touch_type = label_map_Freq[touch_type_idx]  # Get the actual touch type label
        # Store the results
        time_sec = int(rospy.get_time())
        results.append([time_sec, touch_type])

    # MultiClassifier interface
    if MultiClassifier:
        # Convert 5 string touch_type labels back to 2 string contact labels (Contact/NoContact)
        label_map_MC = { 0:"NoContact",1:"Contact", 2:"Contact", 3:"Contact", 4:"Contact"}
        touch_type = label_map_MC[touch_type_idx]
        # Convert 5 string touch_type labels back to 2 int contact idx (Contact->1/NoContact->-1)
        contact_idx_map_MC = { 0:-1,1:1, 2:1, 3:1, 4:1}
        Contact_idx = contact_idx_map_MC[touch_type_idx]
        # initiallize the prediction dict for multiple classifier and default prediction
        prediction = -1
        prediction_dict = {}
        gesture_raw_output_dict = {}
        # Apply multi classifier when contact
        if Contact_idx == 1:
            # Iterate over all classifier and get their result
            for gesture_classifier in gesture_list:
                gesture_prediction,gesture_raw_output = gesture_classifier.gesture_model.single_predict(window.flatten())
                # print([gesture_prediction,gesture_raw_output])
                prediction_dict[gesture_classifier.gesture_name] = gesture_prediction
                if gesture_prediction: #== 1:#gesture_raw_output>0.00001:
                    gesture_raw_output_dict[gesture_classifier.gesture_name] = gesture_raw_output
        print(f"gesture_raw_output_dict: {gesture_raw_output_dict}")

        percentage_dict_val = np.array(list(gesture_raw_output_dict.values()))
        exp_x = np.exp(percentage_dict_val)
        softmax_value = exp_x/np.sum(exp_x)
        soft_dict = {key:softmax_value[i] for i,key in enumerate(gesture_raw_output_dict)}
        print(f"softmax_value: {soft_dict}")

        max_key = None
        for k,v in soft_dict.items():
            if (v == max(soft_dict.values())):
                max_key = k
                # print(f"max_key: {max_key}")
        # final result is {max_key} 

            


    # Log prediction
    detection_duration  = rospy.get_time() - start_time
    if MultiClassifier:
        multiclassifier_output_pre = ','.join([f'{k}: {v}' for k, v in prediction_dict.items()])
        multiclassifier_output_softmax = ','.join([f'{k}: {v}' for k, v in soft_dict.items()])
        # final_result_key = max(percentage_dict,key=percentage_dict.get)
        rospy.loginfo(f'Contact:{touch_type}, Predicted touch_type: {multiclassifier_output_pre} ,\n Percentage: {multiclassifier_output_softmax},final result is {max_key} and the detection duration is {detection_duration}')
    else:
        rospy.loginfo(f'Predicted touch_type: {touch_type} and the detection duration is {detection_duration}')
    
    # Calculate time info
    start_time = np.array(start_time).tolist()
    time_sec = int(start_time)
    time_nsec = start_time-time_sec
    # model_msg.data = np.append(np.array([time_sec-big_time_digits, time_nsec, detection_duration, touch_type_idx], dtype=np.complex128), np.hstack(window))
    model_msg.data = np.array([time_sec-big_time_digits, time_nsec, detection_duration, touch_type_idx], dtype=np.complex128)
    # publish the model message
    # print(model_msg)
    model_pub.publish(model_msg)

if __name__ == "__main__":
    global publish_output, big_time_digits

    # Load inverse label map for decoding predictions
    if classes_num == 5:
        label_map_inv = {0:"NC",1:"ST", 2:"DT", 3:"P", 4:"G"}
        # label_classes_RNN = { 0:"NC",1:"ST", 2:"DT", 3:"P", 4:"G"}
        # label_classes_TCNN = { 0:"NC",1:"ST", 2:"DT", 3:"P", 4:"G"}
        # label_classes_Freq = { 0:"NC",1:"ST", 2:"DT", 3:"P", 4:"G"}
    elif classes_num == 4:
        label_map_inv = {0:"NC",1:"ST", 2:"P", 3:"G"}
        # label_classes_RNN = { 0:"NC",1:"ST", 2:"P", 3:"G"}
        # label_classes_TCNN = { 0:"NC",1:"ST", 2:"P", 3:"G"}
        # label_classes_Freq = { 0:"NC",1:"ST", 2:"P", 3:"G"}
    else:   
        print("Wrong Class Number in Main")
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