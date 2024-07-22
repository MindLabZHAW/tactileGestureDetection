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
	
	$HOME/miniconda/envs/frankapyenv/bin/python3 tactileGestureDetection/main.py

# to chage publish rate of frankastate go to : 
sudo nano /franka-interface/catkin_ws/src/franka_ros_interface/launch/franka_ros_interface.launch
"""

## import required libraries 
import os
import numpy as np
import pandas as pd
import time

# import torch
# from torchvision import transforms

import rospy
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from frankapy import FrankaArm
from franka_interface_msgs.msg import RobotState
from threading import Thread
from threading import Event


# Set the main path
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'

# Define paths for joint motion data
joints_data_path = main_path + 'frankaRobot/robotMotionPoints/robotMotionJointData.csv'


model_msg = Floats()


# Callback function for contact detection
def contact_detection(data):
	# global window
	start_time = rospy.get_time()


	detection_duration  = rospy.get_time()-start_time

	rospy.loginfo('detection duration: %f',detection_duration)



if __name__ == "__main__":
	global publish_output
	event = Event()
	# create robot controller instance
	fa = FrankaArm()
	# subscribe robot data topic for contact detection module
	rospy.Subscriber(name= "/robot_state_publisher_node_1/robot_state",data_class= RobotState, callback =contact_detection)#, callback_args=update_state)#,queue_size = 1)
	rospy.spin()
	
	




