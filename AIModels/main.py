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
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP

from Process_Data.Data2Models import create_tensor_dataset_without_torque

num_features = 4
num_class = 5



class Sequence(nn.Module):
    def __init__(self,network_type) :
        super(Sequence, self).__init__()
        if network_type == 'LSTM':
            num_features = 4
            hidden_size = 50
            self.innernet = nn.LSTM(input_size=num_features*28, hidden_size=hidden_size, num_layers=1, batch_first=True)
        elif network_type == 'GRU':
            num_features = 4
            hidden_size = 50
            self.innernet = nn.GRU(input_size=num_features*28, hidden_size=hidden_size, num_layers=1, batch_first=True)
        elif network_type == 'FCLTC':
            units = 50
            self.innernet = LTC(input_size=num_features*28, units=units, batch_first=True)
        elif network_type == 'FCCfC':
            units = 50
            self.innernet = CfC(input_size=num_features*28, units=units, batch_first=True)
        elif network_type == 'NCPLTC':
            units = 50
            input_size = num_features*28
            output_sizt = 1
            self.innernet = LTC(input_size=input_size, units=AutoNCP(units=units, output_size=output_sizt), batch_first=True)
        elif network_type == 'NCPCfC':
            units = 50
            input_size = num_features*28
            output_sizt = 1
            self.innernet = CfC(input_size=input_size, units=AutoNCP(units=units, output_size=output_sizt), batch_first=True)
        self.linear = nn.Linear(in_features=50, out_features=num_class)

    def forward(self, input):
        x,_ = self.innernet(input)
        x = x[:,-1,:]
        x = self.linear(x)
        return x
    
def get_output(data_ds, model): ## 还没搞完
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

if __name__ == '__main__':  ### 还没搞完
    # set random seed to 0
    torch.manual_seed(2020)
    np.random.seed(2020)
    random.seed(2020)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.get_device_name()
    
    # Load data and create training and testing sets
    training_data = create_tensor_dataset_without_torque(main_path+'/dataset/realData/contact_detection_train.csv',num_classes=num_classes, collision=collision, localization= localization, num_features_lstm=num_features_lstm)
    testing_data = create_tensor_dataset_without_torque(main_path+'/dataset/realData/contact_detection_test.csv',num_classes=num_classes, collision=collision, localization= localization,num_features_lstm=num_features_lstm)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle= True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle= True)
    
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # Build the model
    model= Sequence(num_classes, network_type, num_features_lstm)
    model = model.double()
    # Use Adam optimizer and CrossEntropyLoss as the loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    # Training loop
    for epoch in range(n_epochs):
        running_loss = []
        for X_batch, y_batch in train_dataloader:
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





