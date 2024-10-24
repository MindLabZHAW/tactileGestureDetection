
load gesture_dict

# Initialize a list to store the results
results = []
# Create message for publishing model output (will be used in saceDataNode.py)
model_msg = Floats()

window = np.zeros([window_length, features_num * dof])
print(f'{method}-{model.network_type}\'s window size is {window.shape}')

def contact_multiclassifier(data):
    global window, results, big_time_digits

    start_time = rospy.get_time()

    # Prepare data as done in training
    e = np.array(data.q_d) - np.array(data.q)
    # print("e is ", e)
    de = np.array(data.dq_d) - np.array(data.dq)
    tau_J = np.array(data.tau_J) 
    tau_ext = np.array(data.tau_ext_hat_filtered) 

    new_row = np.column_stack((e,de,tau_J,tau_ext)).reshape(1, features_num * dof)
    # print(f"new row is {new_row}")
    window = np.append(window[1:, :], new_row, axis=0)
    
    # STFT
    fs = 200
    nperseg = 16
    noverlap = nperseg - 1
    data_matrix = [] 
    for feature_idx in range(window.shape[1]):
        signal = window[:, feature_idx]
        if Normalization:
            signal_mean = np.mean(signal)
            signal_std = np.std(signal)
            signal = (signal - signal_mean) / (signal_std + 1e-5)
        # f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap, window=sg.windows.general_gaussian(64, p=1, sig=7))
        f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap, window='hamming')
        data_matrix.append(np.abs(Zxx))

    data_matrix -> data_feature

    similarity_dict = {}
    output_dict = {}
    for gesture in gesture_dict:
        similarity = rbf(gesture.gesture_feature, data_feature)
        similarity_dict[gesture] = similarity
        if similarity >= 90:
            output_dict[gesture] = 1
        else:
            output_dict[gesture] = 0
    
    print(f"Possible gesture is {}" all1gestre)
