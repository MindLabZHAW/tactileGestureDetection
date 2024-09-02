import numpy as np
import time

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

num_classes = 5
time_window = 200

batch_size = 64
lr = 0.001
n_epochs = 100
network_type = '3LCNN'
train_all_data = False


class CNNSequence(nn.Module):
    def __init__(self, network_type):
        super(CNNSequence, self).__init__()
        if network_type == '3LCNN':
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)

    def forward(self, input):
        x = nn.functional.relu(self.conv1(input))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def get_output(data_ds, model):
    labels_pred = []
    model.eval()
    with torch.no_grad():
        for i in range(len(data_ds.data_target)):
            x, y = data_ds.__getitem__(i)
            x = model(x)
            x.squeeze()
            labels_pred.append(x.detach().numpy())
    #convert list type to array
    labels_pred = np.array(labels_pred)
    labels_pred = labels_pred.argmax(axis=1)
    labels_true = np.array(data_ds.data_target[:])
    labels_true = labels_true.astype('int64')

    return torch.tensor(labels_pred), torch.tensor(labels_true)

    

if __name__ == '__main__':
    # NPZ Rawu Data Loading
    loaded_data = np.load('DATA/STFT_images/stft_matrices.npz', allow_pickle=True)    
    stft_matrices = np.array(loaded_data['stft_matrices'])
    labels = loaded_data['labels']
    block_ids = loaded_data['block_ids']
    train_matrices, test_matrices, train_label, test_label = train_test_split(stft_matrices, labels, test_size=0.2, random_state=2024)
    print(train_matrices)
    print(train_matrices.shape)

    torch.manual_seed(2024)
    np.random.seed(2020)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Train & Test Data Loading
    training_data = create_tensor_dataset_stft(stft_matrices=train_matrices, labels=train_label)
    testing_data = create_tensor_dataset_stft(stft_matrices=test_matrices, labels=test_label)

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # Build the model
    model = CNNSequence(network_type)
    model = model.double()
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
            running_loss.append(loss.cpu().detach.numpy())
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
            print("on the test set: \n",confusionMatrix(y_test , y_pred))

            y_pred, y_train = get_output(training_data, model)
            print("on the train set: \n",confusionMatrix(y_train , y_pred))

            #plot confusion matrix using seabon
            confusionMatrixPlot = confusionMatrix.compute().numpy()
            plt.figure()
            label_classes = {0:"ST", 1:"DT", 2:"P", 3:"G", 4:"NC"}
            sns.heatmap(confusionMatrixPlot,annot=True,fmt= 'd',cmap='Blues', xticklabels=label_classes, yticklabels=label_classes)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.show()

        # Save model
        named_tuple = time.localtime() 
        if input('do you want to save the data in trained models? (y/n):')=='y':
            output_path = path_name + network_type + str(time.strftime("_%m_%d_%Y_%H-%M-%S", named_tuple)) + '.pth'

            torch.save({"model_state_dict": model.state_dict(),
                    "optimzier_state_dict": optimizer.state_dict(), "network_type": network_type,
                    "n_epochs": n_epochs , "batch_size": batch_size, "num_features": num_features,
                    "num_classes": num_classes, "lr": lr}, output_path)
            
            print('model is saved successfully!')

    