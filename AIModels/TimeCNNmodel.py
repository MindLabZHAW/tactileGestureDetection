# By Deqing Song

# conda activate frankapyenv
# $HOME/miniconda/envs/frankapyenv/bin/python3 AIModels/main.py

import os
import numpy as np
import pandas as pd
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP

import seaborn as sns
import matplotlib.pyplot as plt

from Process_Data.Data2Models import create_tensor_dataset_tcnn

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
path_name = os.path.dirname(os.path.abspath(__file__))+'/TrainedModels/'

num_features = 4
num_classes = 5
time_window = 28

batch_size = 64
lr = 0.001
n_epochs = 40

network_type = '1L3DTCNN'
train_all_data = False # train a model using all avaiable data
normalization = False

class Time3DCNNSequence(nn.Module):
    def __init__(self, network_type, num_classes=5, num_features=4, time_window=28) :
        super(Time3DCNNSequence, self).__init__()

        if network_type == '1L3DTCNN':
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(28, 3, 3), stride=1, padding=0)
            
            # 定义 3D 池化层
            self.global_max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        
            self.flatten = nn.Flatten() # with batch so flatten from dimension 1 not 0
            self.fc = nn.Linear(32, num_classes)
        elif network_type =='2L3DTCNN':     
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=0)
            self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5, 1, 1), stride=1, padding=0) 
            
            # 定义 3D 池化层
            self.global_max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        
            self.flatten = nn.Flatten() # with batch so flatten from dimension 1 not 0
            self.fc = nn.Linear(32, num_classes)

        self.network_type = network_type

    def forward(self, input):
  
        if self.network_type == '1L3DTCNN':
            x = input.unsqueeze(1)
            x = nn.functional.relu(self.conv1(x))
            # print("After conv1:", x.shape)  # 检查形状
            # x = nn.functional.relu(self.conv2(x))
            # print("After conv2:", x.shape)  # 检查形状
            x = self.global_max_pool(x)
            # print("After MP1:", x.shape)  # 检查形状
            x = self.flatten(x)
            # x = x.view(x.size(0), -1)
            # print("After Flatten:", x.shape)  # 检查形状
            x = self.fc(x)
        if self.network_type == '2L3DTCNN':
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
        for i in range(len(data_ds.data_target)):
            x , y = data_ds.__getitem__(i)
            # print(x.shape)
            x = x.unsqueeze(0)
            # print(x.shape)
            x = model(x)
            x = x.squeeze()
            labels_pred.append(x.detach().numpy())
    #convert list type to array
    labels_pred = np.array(labels_pred)
    labels_pred = labels_pred.argmax(axis=1)
    labels_true = np.array(data_ds.data_target[:])
    labels_true = labels_true.astype('int64')

    return torch.tensor(labels_pred), torch.tensor(labels_true)

if __name__ == '__main__':
    # set random seed to 0
    torch.manual_seed(2020)
    np.random.seed(2020)
    random.seed(2020)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if device.type == "cuda":
        torch.cuda.get_device_name()
    
    # Load data and create training and testing sets
    # training_data = create_tensor_dataset_without_torque('../contactInterpretation-main/dataset/realData/contact_detection_train.csv',num_classes=num_classes, collision=collision, localization= localization, num_features=num_features)
    # testing_data = create_tensor_dataset_without_torque('../contactInterpretation-main/dataset/realData/contact_detection_test.csv',num_classes=num_classes, collision=collision, localization= localization,num_features=num_features)
    training_data = create_tensor_dataset_tcnn(main_path + 'DATA/6_labeled_window_dataset_train_Normalization.csv')
    testing_data = create_tensor_dataset_tcnn(main_path + 'DATA/6_labeled_window_dataset_test_Normalization.csv')

    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle= True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle= True)
    
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # Build the model  
    model= Time3DCNNSequence(network_type)
    model = model.double() # Used to keep the input as double type 
    # Use Adam optimizer and CrossEntropyLoss as the loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()


    # Training loop
    for epoch in range(n_epochs):
        running_loss = []
        for X_batch, y_batch in train_dataloader:
            # print(X_batch.shape)
            # print(X_batch)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            #torch.argmax(y_pred, dim=1)
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

    # Validation
    model.eval()
    
    with torch.no_grad():
        confusionMatrix = ConfusionMatrix(task = "multiclass", num_classes= num_classes)

        y_pred, y_test = get_output(testing_data, model)
        print('y_pred ',y_pred)
        print('y_test ',y_test)
        print("on the test set: \n",confusionMatrix(y_test , y_pred))

        y_pred, y_train = get_output(training_data, model)
        print("on the train set: \n",confusionMatrix(y_train , y_pred))

        #plot confusion matrix using seabon
        confusionMatrixPlot = confusionMatrix.compute().numpy()
        plt.figure()
        label_classes = label_classes = ["NC", "ST", "DT", "P", "G"]
        sns.heatmap(confusionMatrixPlot,annot=True,fmt= 'd',cmap='Blues', xticklabels=label_classes, yticklabels=label_classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
    
    # Save model
    named_tuple = time.localtime() 
    if input('do you want to save the data in trained models? (y/n):')=='y':
        tag = input('Please put a tag if needed: ')
        output_path = path_name + network_type + str(time.strftime("_%m_%d_%Y_%H-%M-%S", named_tuple)) + tag+ '.pth'

        torch.save({"model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict(), "network_type": network_type,
                "n_epochs": n_epochs , "batch_size": batch_size, "num_features": num_features,
                "num_classes": num_classes, "lr": lr}, output_path)
        
        print('model is saved successfully!')



