from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
from collections import Counter
import torch
import os
import joblib


# Load data
data_path = '../DATA/labeled_window_dataset.csv'
df = pd.read_csv(data_path)

# Define columns corresponding to each joint
joint_columns = {
    0: ['e0', 'de0', 'tau_J0', 'tau_ext0'],
    1: ['e1', 'de1', 'tau_J1', 'tau_ext1'],
    2: ['e2', 'de2', 'tau_J2', 'tau_ext2'],
    3: ['e3', 'de3', 'tau_J3', 'tau_ext3'],
    4: ['e4', 'de4', 'tau_J4', 'tau_ext4'],
    5: ['e5', 'de5', 'tau_J5', 'tau_ext5'],
    6: ['e6', 'de6', 'tau_J6', 'tau_ext6'],
}

# Initialize feature and label lists
X_list = []
y_list = []

# Group data by 'block_id'
grouped = df.groupby('window_id')

# Process each group
for window_id, group in grouped:
    window_features = []
    for joint, cols in joint_columns.items():
        joint_data = group.loc[:, cols].values.flatten()
        window_features.extend(joint_data)

    X_list.append(window_features)
    y_list.append(group['window_touch_type'].iloc[0])

# Convert lists to numpy arrays
X = np.array(X_list)
y = np.array(y_list)

# Encode labels
label_classes = {"NC": 0, "ST": 1, "DT": 2, "P": 3, "G": 4}
label_map = {key: value for key, value in label_classes.items()}
y_encoded = np.array([label_map[key] for key in y])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
print("Training set distribution:", Counter(y_resampled))

def dtw_distance(x, y):
    N, M = len(x), len(y)
    dtw_matrix = torch.full((N+1, M+1), float('inf'), device='cuda')
    dtw_matrix[0, 0] = 0

    for i in range(1, N+1):
        for j in range(1, M+1):
            cost = torch.abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + torch.min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    return dtw_matrix[N, M]

# K-NN classifier using DTW
def knn_classify(X_train, y_train, X_test, k):
    y_pred = []
    for test_sample in X_test:
        distances = []
        test_sample_tensor = torch.tensor(test_sample, device='cuda', dtype=torch.float32)
        for train_sample in X_train:
            train_sample_tensor = torch.tensor(train_sample, device='cuda', dtype=torch.float32)
            distance = dtw_distance(test_sample_tensor, train_sample_tensor)
            distances.append(distance.item())
        distances = torch.tensor(distances, device='cuda')
        # Get the k-nearest neighbors
        _, indices = torch.topk(distances, k, largest=False)
        nearest_labels = y_train[indices.cpu()]
        # Majority vote
        prediction = Counter(nearest_labels).most_common(1)[0][0]
        y_pred.append(prediction)
    return np.array(y_pred)

# Predict
y_pred = knn_classify(X_resampled, y_resampled, X_test, 5)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save the trained KNN model for later use
folder_path = 'AIModels/TrainedModels/'
os.makedirs(folder_path, exist_ok=True)
model_path = folder_path + 'KNN_DWT.pkl'
joblib.dump(knn_classify, model_path)
print(f'Model saved to {model_path}')