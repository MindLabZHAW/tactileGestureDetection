# KNN model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib 
import os

# path to save trained models
data_path = 'DATA/tactile_dataset_block.csv'
df = pd.read_csv(data_path)



# Group by 'block_id' and calculate mean of each feature

numerical_columns = df.drop(=["index","time","label","touch_type"]).columns
grouped_df = df.groupby('block_id')[numerical_columns].mean()

# Extract labels
labels = df.groupby('block_id')['touch_type'].first()

# Prepare feature matrix X and label vector y
X = grouped_df.drop(columns=['block_id']).values  # Drop 'block_id' from features
y = labels.values

# Encode labels
label_classes = np.unique(y)
label_map = {label: idx for idx, label in enumerate(label_classes)}
y_encoded = np.array([label_map[label] for label in y])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Initialize KNN classifier
knn = KNeighborsClassifier()

# Define hyperparameters grid
param_grid = {
    'n_neighbors': list(range(1, 21)),  # 搜索1到20范围内的邻居数
    'weights': ['uniform', 'distance']  # 搜索'weights'参数中的两种不同模式
}


# Initialize GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
# print(f'Best parameters: {best_params}')



# Train KNN classifier with the best parameters
best_knn = grid_search.best_estimator_

# Save the trained KNN model for later use
folder_path = 'AIModels/TrainedModels/'
os.makedirs(folder_path, exist_ok=True)
model_path = folder_path  +  'trained_knn_model.pkl'
joblib.dump(best_knn, model_path)
print(f'Model saved to {model_path}')

# # Perform cross-validation on the training set
# cv_scores = cross_val_score(best_knn,X_train,y_train,cv = 5)

# # Predict
# y_pred = best_knn.predict(X_test)
# x_temp = [1.5800049300305543,-0.1132444739341735,-17.335622787475586,-1.891367793083191,17.736581802368164,0.410483717918396,2.499279022216797,-0.0999257862567901,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0372522467744296,0.1544631542649355,-0.1885553765690959,0.0061587490864882,0.02111635161377,0.1464784342336267,-0.0821320042837524,-1.8830910129421636,-0.1403315473000208,0.5157931913428841,-2.673449012281833,0.1833513857656054,2.5995610360039607,0.9506861179069428,-1.8830015233952444,-0.1403389378745897,0.5158180609985245,-2.673378670485581,0.1832823554716121,2.599604072029283,0.950686113546292,-0.0001860987989787,-0.0003809868003953,-0.0013967747863216,-0.0002630234226706,-4.251183452002199e-05,-0.0082523966219078,-0.0003621679285794,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8.948954691900823e-05,-7.3905745689162305e-06,2.4869655640302742e-05,7.034179625264869e-05,-6.903029399329963e-05,4.303602532207407e-05,-4.360650729395843e-09,0.0001860987989787,0.0003809868003953,0.0013967747863216,0.0002630234226706,4.251183452002199e-05,0.0082523966219078,0.0003621679285794,0.1132444739341735,17.335622787475586,1.891367793083191,-17.736581802368164,-0.410483717918396,-2.499279022216797,0.0999257862567901,0,1]
# y_temp = best_knn.predict([x_temp])
# print(y_temp)

# # Evaluate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')

# # Print predicted and true labels
# print(f'Predicted labels: {y_pred}')
# print(f'True labels: {y_test}')

# # Display classification report
# print(classification_report(y_test, y_pred, target_names=label_classes))

# conf_matrix = confusion_matrix(y_test,y_pred)

# #plot confusion matrix using seabon
# plt.figure()
# sns.heatmap(conf_matrix,annot=True,fmt= 'd',cmap='Blues', xticklabels=label_classes, yticklabels=label_classes)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()

