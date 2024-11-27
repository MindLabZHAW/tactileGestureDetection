# Import model
import numpy as np
import torch
import torch.nn as nn
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP

import sys



class RNNSequence(nn.Module):
    def __init__(self, network_type, num_classes = 5, num_features=4, time_window=200) :
        super(RNNSequence, self).__init__()
        if network_type == 'LSTM':
            hidden_size = 50
            self.innernet = nn.LSTM(input_size=num_features * time_window, hidden_size=hidden_size, num_layers=1, batch_first=True)
        elif network_type == 'GRU':
            hidden_size = 50
            self.innernet = nn.GRU(input_size=num_features * time_window, hidden_size=hidden_size, num_layers=1, batch_first=True)
        elif network_type == 'FCLTC':
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
        ## need to check output_size

        self.network_type = network_type
        self.num_classes = num_classes

    def forward(self, input):
        x,_ = self.innernet(input)
        x = x[:,-1,:]
        x = self.linear(x)
        return x
    

class CNNSequence(nn.Module):
    def __init__(self, network_type, num_classes):
        super(CNNSequence, self).__init__()
        if network_type == '2LCNN':
            self.conv1 = nn.Conv2d(in_channels=28, out_channels=16, kernel_size=3, stride=1, padding=0)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
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

    def forward(self, input):
        if self.network_type == '2LCNN':
            x = nn.functional.relu(self.conv1(input))
            # print("After conv1:", x.shape)  # Check the shape
            x = nn.functional.max_pool2d(x, (1,2))
            # print("After MP1:", x.shape)  # Check the shape
            x = nn.functional.relu(self.conv2(x))
            # print("After conv2:", x.shape)  # Check the shape
            x = nn.functional.max_pool2d(x, (1,2))
            # print("After MP1:", x.shape)  # Check the shape
            x = torch.flatten(x) # without batch so flatten from dimension 0
            # print("After Flatten:", x.shape)  # Check the shape
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
        elif self.network_type == '3LCNN':
            x = nn.functional.relu(self.conv1(input))
            # print("After conv1:", x.shape)  # Check the shape
            x = nn.functional.avg_pool2d(x, (2, 2))
            # print("After MP1:", x.shape)  # Check the shape
            x = nn.functional.relu(self.conv2(x))
            # print("After conv2:", x.shape)  # Check the shape
            x = nn.functional.avg_pool2d(x, (2, 1))
            # print("After MP2:", x.shape)  # Check the shape
            x = nn.functional.relu(self.conv3(x))
            # print("After conv3:", x.shape)  # Check the shape
            x = nn.functional.avg_pool2d(x, (2, 1))
            # print("After MP3:", x.shape)  # Check the shape
            x = self.flatten(x)
            # x = x.view(x.size(0), -1)
            # print("After Flatten:", x.shape)  # Check the shape
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
        x = input.unsqueeze(1)
        x = nn.functional.relu(self.conv1(x))
        # print("After conv1:", x.shape)  # Check the shape
        x = nn.functional.relu(self.conv2(x))
        # print("After conv2:", x.shape)  # Check the shape
        x = self.global_max_pool(x)
        # print("After MP1:", x.shape)  # Check the shape
        x = self.flatten(x)
        # x = x.view(x.size(0), -1)
        # print("After Flatten:", x.shape)  # Check the shape
        x = self.fc(x)
        return x


class Time3DCNNSequence(nn.Module):
    def __init__(self, network_type, num_classes=5, num_features=4, time_window=28) :
        super(Time3DCNNSequence, self).__init__()
        if network_type == '1L3DTCNN':
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(28, 3, 3), stride=1, padding=0)
            
            # Define 3D Pooling layer
            self.global_max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        
            self.flatten = nn.Flatten() # with batch so flatten from dimension 1 not 0
            self.fc = nn.Linear(32, num_classes)
        elif network_type =='2L3DTCNN':     
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=0)
            self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5, 1, 1), stride=1, padding=0) 
            
            # Define 3D Pooling layer
            self.global_max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        
            self.flatten = nn.Flatten() # with batch so flatten from dimension 1 not 0
            self.fc = nn.Linear(32, num_classes)

        self.network_type = network_type
        self.num_classes = num_classes
        ## need to check output_size

    def forward(self, input):
        if self.network_type == '1L3DTCNN':
            x = input.unsqueeze(1)
            x = nn.functional.relu(self.conv1(x))
            # print("After conv1:", x.shape)  # Check the shape
            #x = nn.functional.relu(self.conv2(x))
            # print("After conv2:", x.shape)  # Check the shape
            x = self.global_max_pool(x)
            # print("After MP1:", x.shape)  # Check the shape
            x = self.flatten(x)
            # x = x.view(x.size(0), -1)
            # print("After Flatten:", x.shape)  # Check the shape
            x = self.fc(x)
        if self.network_type == '2L3DTCNN':
            x = input.unsqueeze(1)
            x = nn.functional.relu(self.conv1(x))
            # print("After conv1:", x.shape)  # Check the shape
            x = nn.functional.relu(self.conv2(x))
            # print("After conv2:", x.shape)  # Check the shape
            x = self.global_max_pool(x)
            # print("After MP1:", x.shape)  # Check the shape
            x = self.flatten(x)
            # x = x.view(x.size(0), -1)
            # print("After Flatten:", x.shape)  # Check the shape
            x = self.fc(x)
        return x

def import_rnn_models(PATH:str, network_type:str, num_classes:int,  num_features:int, time_window:int):

	model = RNNSequence(network_type = network_type, num_classes = num_classes, num_features=num_features, time_window=time_window)
	checkpoint = torch.load(PATH)
	model.load_state_dict(checkpoint["model_state_dict"])
	
	print('***  Models loaded  ***')
	return model.eval()

def import_cnn_models(PATH:str, network_type:str, num_classes:int):
    
    if network_type in ['2LCNN', '3LCNN']:
        model = CNNSequence(network_type = network_type, num_classes = num_classes)
    elif network_type in ['2L3DCNN', 'T2L3DCNN']:
        model = CNNSequence3D(network_type = network_type, num_classes = num_classes)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print('***  Models loaded  ***')
    return model.eval()

def import_tcnn_models(PATH:str, network_type:str, num_classes:int,  num_features:int, time_window:int):
    model = Time3DCNNSequence(network_type = network_type, num_classes = num_classes, num_features=num_features, time_window=time_window)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()