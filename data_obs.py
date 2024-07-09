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


# CREATE FOLDER FOR KEEP DATA
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir ,"data")
#print(os.path)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Save data to csv file

def save_data_to_csv(data,attribute_name):
    value = getattr(data,attribute_name)
    filepath = os.path.join(data_dir,attribute_name + '.csv')
    with open(filepath,mode = 'w') as file:
        writer = csv.writer(file)
        if isinstance(value,(tuple,list)):
            writer.writerow(value)
        else:
            writer.writerows([value])

def print_robot_state(data):
    # rospy.loginfo(np.array(data.q_d) - np.array(data.q))
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.q)
    # print(f"the type of dq is {type(data.dq)}")
    # print(f"the type of elbow is {type(data.elbow)}")
    
    attributes = ["elbow"]
    for attribute_name in attributes:
        if hasattr(data,attribute_name):
            print(getattr(data,attribute_name))
            save_data_to_csv(data,attribute_name)
    


if __name__ == '__main__':
    # create FrankaArm instance
    fa = FrankaArm()

    rospy.Subscriber(name="/robot_state_publisher_node_1/robot_state", data_class=RobotState, callback=print_robot_state, queue_size=1)
    rospy.spin()