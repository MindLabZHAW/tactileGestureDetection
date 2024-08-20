# By Deqing Song

# conda activate frankapyenv
# $HOME/miniconda/envs/frankapyenv/bin/python3 AIModels/main.py

import os
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP

import sys
sys.path.append("Process_Data")
from Data2Models import create_tensor_dataset

num_features = 1
num_classes = 4

network_type = 'LSTM'
train_all_data = False # train a model using all avaiable data

collision = False; localization = False; n_epochs = 15; batch_size = 64; num_classes = 5; lr = 0.001
#collision = True; localization = False; n_epochs = 120; batch_size = 64; num_classes = 2; lr = 0.001
#collision = False; localization = True; n_epochs = 110; batch_size =64; num_classes = 2; lr = 0.001


class Sequence(nn.Module):
    def __init__(self,network_type) :
        super(Sequence, self).__init__()
        if network_type == 'LSTM':
            num_features = 3
            hidden_size = 50
            self.innernet = nn.LSTM(input_size=num_features*499, hidden_size=hidden_size, num_layers=1, batch_first=True)
        elif network_type == 'GRU':
            num_features = 4
            hidden_size = 50
            self.innernet = nn.GRU(input_size=num_features*28, hidden_size=hidden_size, num_layers=1, batch_first=True)
        elif network_type == 'FCLTC':
            num_features = 4
            units = 50
            self.innernet = LTC(input_size=num_features*28, units=units, batch_first=True)
        elif network_type == 'FCCfC':
            num_features = 4
            units = 50
            self.innernet = CfC(input_size=num_features*28, units=units, batch_first=True)
        elif network_type == 'NCPLTC':
            num_features = 4
            units = 53
            input_size = num_features*28
            output_size = 50
            self.innernet = LTC(input_size=input_size, units=AutoNCP(units=units, output_size=output_size), batch_first=True)
        elif network_type == 'NCPCfC':
            num_features = 4
            units = 53
            input_size = num_features*28
            output_size = 50
            self.innernet = CfC(input_size, AutoNCP(units=units, output_size=output_size), batch_first=True) ### youwenti!
        self.linear = nn.Linear(in_features=50, out_features=num_classes)
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
        for i in range(data_ds.data_target.shape[0]):
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
    training_data = create_tensor_dataset('DATA/tactile_dataset_block_train.csv')
    testing_data = create_tensor_dataset('DATA/tactile_dataset_block_test.csv')

    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle= True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle= True)
    
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # Build the model  
    model= Sequence(network_type)
    model = model.double()
    # Use Adam optimizer and CrossEntropyLoss as the loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    # Training loop
    for epoch in range(n_epochs):
        running_loss = []
        index_i = 0
        for X_batch, y_batch in train_dataloader:
            index_i += 1
            print(index_i)
            print(X_batch.shape)
            print(X_batch)
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
        print("on the test set: \n",confusionMatrix(y_test , y_pred))

        y_pred, y_train = get_output(training_data, model)
        print("on the train set: \n",confusionMatrix(y_train , y_pred))





