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

import sys
sys.path.append("Process_Data")
from Data2Models import create_tensor_dataset

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
path_name = os.path.dirname(os.path.abspath(__file__))+'/TrainedModels/'

num_features = 4
num_classes = 5
time_window = 28

batch_size = 64
lr = 0.001
n_epochs = 100

network_type = 'NCPCfC'
train_all_data = True # train a model using all avaiable data

# collision = False; localization = False; n_epochs = 15; batch_size = 64; num_classes = 5; lr = 0.001
# collision = True; localization = False; n_epochs = 120; batch_size = 64; num_classes = 2; lr = 0.001
# collision = False; localization = True; n_epochs = 110; batch_size =64; num_classes = 2; lr = 0.001

class Sequence(nn.Module):
    def __init__(self, network_type, num_classes=5, num_features=4, time_window=28) :
        super(Sequence, self).__init__()
        if network_type == 'LSTM':
            hidden_size = 50
            self.innernet = nn.LSTM(input_size=num_features * time_window, hidden_size=hidden_size, num_layers=1, batch_first=True)
        elif network_type == 'GRU': # >100
            hidden_size = 50
            self.innernet = nn.GRU(input_size=num_features * time_window, hidden_size=hidden_size, num_layers=1, batch_first=True)
        elif network_type == 'FCLTC': # 50-60
            units = 50
            self.innernet = LTC(input_size=num_features * time_window, units=units, batch_first=True)
        elif network_type == 'FCCfC':
            units = 50
            self.innernet = CfC(input_size=num_features * time_window, units=units, batch_first=True)
        elif network_type == 'NCPLTC':
            units = 53
            input_size = num_features * time_window
            output_size = 50
            self.innernet = LTC(input_size=input_size, units=AutoNCP(units=units, output_size=output_size), batch_first=True)
        elif network_type == 'NCPCfC':
            units = 53
            input_size = num_features * time_window
            output_size = 50
            self.innernet = CfC(input_size, AutoNCP(units=units, output_size=output_size), batch_first=True) ### youwenti!
        self.linear = nn.Linear(in_features=50, out_features=num_classes)
        
        self.network_type = network_type
        ## need to check output_size

    def forward(self, input):
        x,_ = self.innernet(input)
        x = x[:,-1,:]
        x = self.linear(x)
        return x
    
def get_output(data_ds, model): 
    labels_pred = []
    model.eval()
    with torch.no_grad():
        for i in range(len(data_ds.data_target)):
            x , y = data_ds.__getitem__(i)
            x = x[None, :]

            x = model(x)
            x = x.squeeze()
            #labels_pred.append(torch.Tensor.cpu(x.detach()).numpy())
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
    training_data = create_tensor_dataset(main_path + 'DATA/6_labeled_window_dataset_train.csv')
    testing_data = create_tensor_dataset(main_path + 'DATA/6_labeled_window_dataset_test.csv')

    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle= True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle= True)
    
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # Build the model  
    model= Sequence(network_type)
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



