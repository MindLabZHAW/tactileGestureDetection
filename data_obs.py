'''
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
	
	$HOME/miniconda/envs/frankapyenv/bin/python3 frankaRobot/main.py

    $HOME/miniconda/envs/frankapyenv/bin/python3 tactileGestureDetection/test.py
'''

import numpy as np
import pandas as pd
import time
import json

import rospy
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from frankapy import FrankaArm
from franka_interface_msgs.msg import RobotState
from threading import Thread
from threading import Event

import csv
import datetime
import os
import subprocess


# Global constant for attributes
ATTRIBUTES = [
    "elbow", "O_T_EE", "tau_J_d", "q", "q_d", "dq", "tau_J", "dtau_J", "gravity", 
    "coriolis", "O_F_ext_hat_K", "m_ee", "IA", "tau_ext_hat_filtered", "joint_contact", 
    "artesian_contact", "joint_collision", "cartesian_collision"
]

# CREATE FOLDER FOR KEEP DATA
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir ,"data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Create subfolder
subfolder_name = input("Enter the subfolder name: ")
subfolder_path = os.path.join(data_dir,subfolder_name)
if not os.path.exists(subfolder_path):
    os.makedirs(subfolder_path)

# create the name of csv file
start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json_file_path = os.path.join(subfolder_path, start_timestamp + "_all_attributes.json")

# initialize json file
def initialize_json():
    with open(json_file_path, mode="w") as file:
        json.dump([], file)

# Save data to json file
def save_data_to_json(attribute_name, value, timestamp):
    data_entry = {
        "timestamp": timestamp,
        "attribute_name": attribute_name,
        "values": value
    }
    
    with open(json_file_path, mode='r+') as file:
        data = json.load(file)
        data.append(data_entry)
        file.seek(0)
        json.dump(data, file, indent=4)

# Initialize individual csv files
def initialize_csv(attribute_name):
    csv_file_path = os.path.join(subfolder_path, f"{attribute_name}.csv")
    headers = ["timestamp", "values"]
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

# Save data to csv file
def save_data_to_csv(attribute_name,value,timestamp):
    csv_file_path = os.path.join(subfolder_path, f"{attribute_name}.csv")
    
    with open(csv_file_path, mode='a', newline="") as file:
        writer = csv.writer(file)
        if isinstance(value, (tuple, list)):
            row = [timestamp] + list(value)
        else:
            row = [timestamp, value]
        writer.writerow(row)
    

def print_robot_state(data):
    timestamp = datetime.datetime.now().isoformat()
    data_entry = {}

    for attribute_name in ATTRIBUTES:
        if hasattr(data,attribute_name):
            value = getattr(data,attribute_name)
            data_entry[attribute_name] = value
            save_data_to_json(attribute_name, value, timestamp)
            save_data_to_csv(attribute_name, value, timestamp)
    

    print(f"{timestamp} : {data_entry}")


if __name__ == '__main__':
    initialize_json()
    for attribute in ATTRIBUTES:
        initialize_csv(attribute)
    # create FrankaArm instance
    fa = FrankaArm()

    rospy.Subscriber(name="/robot_state_publisher_node_1/robot_state", data_class=RobotState, callback=print_robot_state, queue_size=1)
    rospy.spin()

    # call visualization script
    #subprocess.run(["python","visualization.py",csv_file_path])

