import numpy as np
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append("Process_Data")
from Data2Models import create_tensor_dataset_stft

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
path_name = os.path.dirname(os.path.abspath(__file__))+'/TrainedModels/'

num_classes = 5
time_window = 28

batch_size = 64
lr = 0.001
n_epochs = 150
network_type = '2L3DCNN'
train_all_data = False

class CNNSequence(nn.Module):
    def __init__(self,network_type, num_classes):
        super(CNNSequence, self).__init__()

        if network_type == '2LCNN':
            self.conv1 = nn.Conv2d(in_channels=28, out_channels=16, kernel_size=3, stride=1, padding=0)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
            # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
            self.flatten = nn.Flatten() # with batch so flatten from dimension 1 not 0
            self.fc1 = nn.Linear(32 * 5 * 5, 64)
            self.fc2 = nn.Linear(64, num_classes)
        elif network_type == '3LCNN':
            self.conv1 = nn.Conv2d(in_channels=28, out_channels=16, kernel_size=3, stride=1, padding=0)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
            self.flatten = nn.Flatten() # with batch so flatten from dimension 1 not 0
            self.fc1 = nn.Linear(64* 14 * 9, 128)
            self.fc2 = nn.Linear(128, num_classes)

        self.network_type = network_type
        self.num_classes = num_classes

    def forward(self,x):

        if self.network_type == '2LCNN':
            x = nn.functional.relu(self.conv1(x))
            # print("After conv1:", x.shape)  # 检查形状
            x = nn.functional.max_pool2d(x, (1, 2))
            # print("After MP1:", x.shape)  # 检查形状
            x = nn.functional.relu(self.conv2(x))
            # print("After conv2:", x.shape)  # 检查形状
            x = nn.functional.max_pool2d(x, (1, 2))
            # print("After MP1:", x.shape)  # 检查形状
            # x = nn.functional.relu(self.conv3(x))
            # x = nn.functional.max_pool2d(x, 2)
            x = self.flatten(x)
            # x = x.view(x.size(0), -1)
            # print("After Flatten:", x.shape)  # 检查形状
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
        elif self.network_type == '3LCNN':
            x = nn.functional.relu(self.conv1(x))
            # print("After conv1:", x.shape)  # 检查形状
            x = nn.functional.avg_pool2d(x, (2, 2))
            # print("After MP1:", x.shape)  # 检查形状
            x = nn.functional.relu(self.conv2(x))
            # print("After conv2:", x.shape)  # 检查形状
            x = nn.functional.avg_pool2d(x, (2, 1))
            # print("After MP2:", x.shape)  # 检查形状
            x = nn.functional.relu(self.conv3(x))
            # print("After conv3:", x.shape)  # 检查形状
            x = nn.functional.avg_pool2d(x, (2, 1))
            # print("After MP3:", x.shape)  # 检查形状
            x = self.flatten(x)
            # x = x.view(x.size(0), -1)
            # print("After Flatten:", x.shape)  # 检查形状
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)

        return x
    
class CNNSequence3D(nn.Module):
    def __init__(self, network_type, num_classes):
        super(CNNSequence3D, self).__init__()

        if network_type in ['2L3DCNN', 'T2L3DCNN']:
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(28, 3, 3), stride=1, padding=0)
            self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=1, padding=0)
            self.global_max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
            self.flatten = nn.Flatten() # with batch so flatten from dimension 1 not 0
            self.fc = nn.Linear(32, num_classes)
        
        self.network_type = network_type
        self.num_classes = num_classes

    def forward(self, input):

        if self.network_type in ['2L3DCNN', 'T2L3DCNN']:
            x = input.unsqueeze(1)
            x = nn.functional.relu(self.conv1(x))
            # print("After conv1:", x.shape)  # 检查形状
            x = nn.functional.relu(self.conv2(x))
            # print("After conv2:", x.shape)  # 检查形状
            x = self.global_max_pool(x)
            # print("After MP1:", x.shape)  # 检查形状
            x = self.flatten(x)
            # x = x.view(x.size(0), -1)
            # print("After Flatten:", x.shape)  # 检查形状
            x = self.fc(x)
        
        return x

def get_output(data_ds, model):
    labels_pred = []
    model.eval()
    with torch.no_grad():
        for i in range(len(data_ds.labels)):
            x, y = data_ds.__getitem__(i)
            # print(x.shape)
            x = x.unsqueeze(0)
            # print(x.shape)
            x = model(x)
            x = x.squeeze()
            labels_pred.append(x.detach().numpy())
  
    #convert list type to array
    labels_pred = np.array(labels_pred)
    labels_pred = labels_pred.argmax(axis=1)
    labels_true = np.array(data_ds.labels[:])
    labels_true = labels_true.astype('int64')

    return torch.tensor(labels_pred), torch.tensor(labels_true)

    

if __name__ == '__main__':
    # NPZ Raw Data Loading
    if network_type in ['2LCNN', '2L3DCNN']:
        loaded_data = np.load('DATA/STFT_images/stft_matrices.npz', allow_pickle=True)    
        stft_matrices = np.array(loaded_data['stft_matrices'])
        labels_str = loaded_data['labels']
        str2int = {"NC": 0, "ST": 1, "DT": 2, "P": 3, "G": 4}
        labels = [str2int[string] for string in labels_str]
        window_ids = loaded_data['window_ids']
        train_matrices, test_matrices, train_labels, test_labels = train_test_split(stft_matrices, labels, test_size=0.2, random_state=2024)
    elif network_type == '3LCNN':
        loaded_data = np.load('DATA/CWT_images/cwt_matrices.npz', allow_pickle=True)    
        cwt_matrices = np.array(loaded_data['cwt_matrices'])
        labels_str = loaded_data['labels']
        str2int = {"NC": 0, "ST": 1, "DT": 2, "P": 3, "G": 4}
        labels = [str2int[string] for string in labels_str]
        window_ids = loaded_data['window_ids']
        train_matrices, test_matrices, train_labels, test_labels = train_test_split(cwt_matrices, labels, test_size=0.2, random_state=2024)
    if network_type == 'T2L3DCNN':
        loaded_data = np.load('DATA/T_images/T_matrices.npz', allow_pickle=True)    
        stft_matrices = np.array(loaded_data['T_matrices'])
        labels_str = loaded_data['labels']
        str2int = {"NC": 0, "ST": 1, "DT": 2, "P": 3, "G": 4}
        labels = [str2int[string] for string in labels_str]
        window_ids = loaded_data['window_ids']
        train_matrices, test_matrices, train_labels, test_labels = train_test_split(stft_matrices, labels, test_size=0.2, random_state=2024)
    # print(train_matrices)

    torch.manual_seed(2024)
    np.random.seed(2020)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Train & Test Data Loading
    training_data = create_tensor_dataset_stft(stft_matrices=train_matrices, labels=train_labels)
    testing_data = create_tensor_dataset_stft(stft_matrices=test_matrices, labels=test_labels)

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # Build the model
    if network_type in ['2LCNN', '3LCNN']:
        model = CNNSequence(network_type=network_type, num_classes=num_classes)
    elif network_type in ['2L3DCNN', 'T2L3DCNN']:
        model = CNNSequence3D(network_type=network_type, num_classes=num_classes)
    # Use Adam optimizer and CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    # Training loop
    for epoch in range(n_epochs):
        running_loss = []
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
        if train_all_data: 
            for X_batch, y_batch in test_dataloader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                #torch.argmax(y_pred, dim=1)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                running_loss.append(loss.cpu().detach().numpy())
        print("Epoch: {}/{} - learning rate: {:.5f}, classification loss: {:.4f}".format(epoch + 1, n_epochs, optimizer.param_groups[0]['lr'], np.mean(running_loss)))
        
    # validation
    model.eval()
    with torch.no_grad():
        confusionMatrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        
        y_pred, y_test = get_output(testing_data, model)
        # print('y_pred ',y_pred)
        # print('y_test ',y_test)
        print("on the test set: \n",confusionMatrix(y_test , y_pred))

        y_pred, y_train = get_output(training_data, model)
        print("on the train set: \n",confusionMatrix(y_train , y_pred))

        #plot confusion matrix using seabon
        confusionMatrixPlot = confusionMatrix.compute().numpy()
        plt.figure()
        label_classes = ["NC", "ST", "DT", "P", "G"]
        sns.heatmap(confusionMatrixPlot,annot=True,fmt= 'd',cmap='Blues', xticklabels=label_classes, yticklabels=label_classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    # Save model
    named_tuple = time.localtime() 
    if input('do you want to save the data in trained models? (y/n):')=='y':
        tag = input('Please put a tag if needed: ')
        output_path = path_name + network_type + str(time.strftime("_%m_%d_%Y_%H-%M-%S", named_tuple)) + tag + '.pth'

        torch.save({"model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict(), "network_type": network_type,
                "n_epochs": n_epochs , "batch_size": batch_size,
                "num_classes": num_classes, "lr": lr}, output_path)
        
        print('model is saved successfully!')

    
