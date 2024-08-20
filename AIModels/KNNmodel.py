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

# path to save trained models
data_path = 'DATA/tactile_dataset_block.csv'
df = pd.read_csv(data_path)


# Group by 'block_id' and calculate mean of each feature
grouped_df = df.groupby('block_id').apply(np.mean)

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
model_path = folder_path  +  'trained_knn_model.pkl'
joblib.dump(best_knn, model_path)
print(f'Model saved to {model_path}')

# # Perform cross-validation on the training set
# cv_scores = cross_val_score(best_knn,X_train,y_train,cv = 5)

# # Predict
# y_pred = best_knn.predict(X_test)

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

