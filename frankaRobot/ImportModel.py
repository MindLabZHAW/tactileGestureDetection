# Import model
import numpy as np
import torch
import torch.nn as nn
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP

import sys



class Sequence(nn.Module):
    def __init__(self, network_type, num_classes = 5, num_features=4, time_window=200) :
        super(Sequence, self).__init__()
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

def import_rnn_models(PATH:str, network_type:str, num_classes:int,  num_features:int, time_window:int):

	model = Sequence(network_type = network_type, num_classes = num_classes, num_features=num_features, time_window=time_window)
	checkpoint = torch.load(PATH)
	model.load_state_dict(checkpoint["model_state_dict"])
	
	print('***  Models loaded  ***')
	return model.eval()