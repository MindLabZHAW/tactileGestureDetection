
""" 
how to run?



 """
import os
import numpy as np
# import pandas as pd
import time

# import torch
# from torchvision import transforms

# import rospy
# from std_msgs.msg import Float64
# from rospy_tutorials.msg import Floats
# from rospy.numpy_msg import numpy_msg
# from frankapy import FrankaArm
# from franka_interface_msgs.msg import RobotState
from threading import Thread
from threading import Event
# from ImportModel import import_Knn_models

# Set the main path
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
print(f"main_path is {main_path}")
# Define parameters for the KNN models
num_neighbors = 1
weights = 'uniform'

contact_detection_path = main_path + ""
