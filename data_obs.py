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

    $HOME/miniconda/envs/frankapyenv/bin/python3 tactileGestureDetection/data_obs.py
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
from threading import Lock, Thread
from threading import Event

import csv
import datetime
import os
import subprocess


# Global constant for attributes
ATTRIBUTES = {
    "tau_J_d": ['tau_J_d0','tau_J_d1', 'tau_J_d2', 'tau_J_d3', 'tau_J_d4', 'tau_J_d5', 'tau_J_d6'],
    "tau_J": ['tau_J0','tau_J1', 'tau_J2', 'tau_J3', 'tau_J4', 'tau_J5', 'tau_J6'],
    "tau_ext_hat_filtered": ['tau_ext0','tau_ext1','tau_ext2','tau_ext3','tau_ext4','tau_ext5','tau_ext6'],
    "q": ['q0','q1','q2','q3','q4','q5','q6'],
    "q_d": ['q_d0','q_d1','q_d2','q_d3','q_d4','q_d5','q_d6'],
    "dq": ['dq0','dq1','dq2','dq3','dq4','dq5','dq6'],
    "dq_d": ['dq_d0','dq_d1','dq_d2','dq_d3','dq_d4','dq_d5','dq_d6'],
}

# Global constants for calculated attributes
CALCULATED_ATTRIBUTES = {
    "e": ['e0','e1','e2','e3','e4','e5','e6'],
    "de": ['de0','de1','de2','de3','de4','de5','de6'],
    "etau": ['etau_J0','etau_J1', 'etau_J2','etau_J3','etau_J4','etau_J5','etau_J6']
}

# CREATE FOLDER FOR KEEP DATA
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Create subfolder
subfolder_name = input("Enter the subfolder name: ")
subfolder_path = os.path.join(data_dir, subfolder_name)
if not os.path.exists(subfolder_path):
    os.makedirs(subfolder_path)

# create the "plot" subfolder
plot_folder_path = os.path.join(subfolder_path, "plot")
if not os.path.exists(plot_folder_path):
    os.makedirs(plot_folder_path)

# create the name of json file
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
        "values": dict(zip(ATTRIBUTES.get(attribute_name, CALCULATED_ATTRIBUTES.get(attribute_name, [])), value))
    }
    
    with open(json_file_path, mode='r+') as file:
        data = json.load(file)
        data.append(data_entry)
        file.seek(0)
        json.dump(data, file, indent=4)

# Initialize CSV files for each attribute
def initialize_csv(attribute_name, headers):
    csv_file_path = os.path.join(subfolder_path, f"{attribute_name}.csv")
    headers = ["timestamp"] + headers
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

# Save data to csv file
def save_data_to_csv(attribute_name, value, timestamp):
    csv_file_path = os.path.join(subfolder_path, f"{attribute_name}.csv")
    
    with open(csv_file_path, mode='a', newline="") as file:
        writer = csv.writer(file)
        if isinstance(value, (tuple, list)):
            row = [timestamp] + list(value)
        else:
            row = [timestamp, value]
        writer.writerow(row)

# Print and save robot state
def print_robot_state(data):
    timestamp = datetime.datetime.now().isoformat()
    data_entry = {}

    for attribute_name, headers in ATTRIBUTES.items():
        if hasattr(data, attribute_name):
            value = getattr(data, attribute_name)
            data_entry[attribute_name] = value
            save_data_to_json(attribute_name, value, timestamp)
            save_data_to_csv(attribute_name, value, timestamp)
            print(f"{timestamp} : {data_entry}")

    # Calculate e, de, etau
    if all(hasattr(data, attr) for attr in ["q_d", "q", "dq_d", "dq", "tau_J_d", "tau_J"]):
        e = np.array(getattr(data, "q_d")) - np.array(getattr(data, "q"))
        de = np.array(getattr(data, "dq_d")) - np.array(getattr(data, "dq"))
        etau = np.array(getattr(data, "tau_J_d")) - np.array(getattr(data, "tau_J"))

        e = e.tolist()
        de = de.tolist()
        etau = etau.tolist()

        save_data_to_json("e", e, timestamp)
        save_data_to_csv("e", e, timestamp)
        save_data_to_json("de", de, timestamp)
        save_data_to_csv("de", de, timestamp)
        save_data_to_json("etau", etau, timestamp)
        save_data_to_csv("etau", etau, timestamp)


def initialize_ros_node():
    try:
        rospy.get_master().getPid()
    except:
        rospy.init_node('franka_data_collector', anonymous=True)


if __name__ == '__main__':
    initialize_json()

    json_lock = Lock()
    csv_lock = Lock()

    for attribute, headers in {**ATTRIBUTES, **CALCULATED_ATTRIBUTES}.items():
        initialize_csv(attribute, headers)

    # Initialize ROS node
    initialize_ros_node()

    # create FrankaArm instance
    fa = FrankaArm()

    rospy.Subscriber(name="/robot_state_publisher_node_1/robot_state", data_class=RobotState, callback=print_robot_state, queue_size=1000)
    rospy.spin()


