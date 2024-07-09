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