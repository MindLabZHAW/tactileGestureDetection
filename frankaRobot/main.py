import os
import numpy as np
import pandas as pd
import joblib
import rospy
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from franka_interface_msgs.msg import RobotState
from threading import Event
from frankapy import FrankaArm

# Set the main path
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
print(f"main_path is {main_path}")

# Parameters for the KNN models
window_length = 200
dof = 7
features_num = 4

# Load the KNN model

model_path = '/home/weimindeqing/contactInterpretation/tactileGestureDetection/AIModels/TrainedModels/trained_knn_model.pkl'
model = joblib.load(model_path)

# Prepare window to collect features
window = np.zeros([window_length, features_num * dof])

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

    # Create new row and update the sliding window
    new_row = np.hstack((tau_J,tau_ext,e_q, e_dq)).reshape((1, features_num * dof))
    print(f"new row is {new_row}")
    window = np.append(window[1:, :], new_row, axis=0)

    # Flatten the window to create a feature vector for the model
    feature_vector = window.mean(axis=0).reshape(1, -1)

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