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

import os
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
from threading import Event

def print_robot_state(data):
    # rospy.loginfo(np.array(data.q_d) - np.array(data.q))
    print(data.q)

# def move_robot(fa:FrankaArm, event: Event):

# 	joints = pd.read_csv(joints_data_path)

# 	# preprocessing
# 	joints = joints.iloc[:, 1:8]
# 	joints.iloc[:,6] -= np.deg2rad(45) 
# 	print(joints.head(5), '\n\n')
# 	fa.goto_joints(np.array(joints.iloc[0]),ignore_virtual_walls=True)
# 	fa.goto_gripper(0.02)
	
# 	while True:	
# 		try:	
# 			for i in range(joints.shape[0]):
# 				fa.goto_joints(np.array(joints.iloc[i]),ignore_virtual_walls=True,duration=4)
# 				#time.sleep(0.01)

# 		except Exception as e:
# 			print(e)
# 			event.set()
# 			break
	
# 	print('fininshed .... !')

if __name__ == '__main__':
    # create FrankaArm instance
    event = Event()
    fa = FrankaArm()
    
    # # create folder
    # folder_name = input('Enter tag name: ')

    # datapath = 'Obs_DATA/' + folder_name

    # os.makedirs(datapath)
    # print("Directory '%s' created" % datapath)

    rospy.Subscriber(name="/robot_state_publisher_node_1/robot_state", data_class=RobotState, callback=print_robot_state, queue_size=1)
    # move_robot(fa, event)