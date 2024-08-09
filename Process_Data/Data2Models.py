import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class create_tensor_dataset_without_torque(Dataset):
    
    def __init__(self, path = '../contactInterpretation-main/dataset/realData/contact_detection_train.csv', transform = transforms.Compose([transforms.ToTensor()]), num_classes =5,
                 num_features_dataset = 28, num_features = 4, data_seq = 28, desired_seq = 28, localization= False, collision =False):
        self.path = path #n * 784
        self.transform = transform
        self.num_features_dataset = num_features_dataset # 4(num_features) * 7(dof) = 28
        self.num_features = num_features # input features number(tau(t), tau_ext(t), e(t), de(t))
        self.data_seq = data_seq
        self.desired_seq = desired_seq
        self.dof = 7
        self.num_classes = num_classes
        self.localization = localization
        self.collision = collision
        if collision and localization:
            print('collision and localization cannot be true at the same time!')
            exit()
            

        self.read_dataset()
        self.data_in_seq()
        
    def __len__(self):
        return len(self.data_target)


    def __getitem__(self, idx: int): # input the inquired line of data (1*784)

        data_sample = torch.tensor(self.data_input.iloc[idx].values)
        data_sample = torch.reshape(data_sample, (self.dof ,self.num_features*self.desired_seq))

        target = self.data_target.iloc[idx]

        return data_sample, target # return a tensor 7*112 and the target integer like 0, 1, 2(-1 is already dropped)


    def read_dataset(self):
        
        labels_map = {
            0: 'Noncontact',
            1: 'Intentional_Link5',
            2: 'Intentional_Link6',
            3: 'Collision_Link5',
            4: 'Collision_Link6',
        }
        # laod data from csv file
        data = pd.read_csv(self.path)
        # specifying target and data
        data_input = data.iloc[:,1:data.shape[1]]
        data_target = data['Var1']
        # print(data_input)
        # print(data_target)

        # changing labels to numbers
        for i in range(data_input.shape[0]):
            for j in range(len(labels_map)):

                if self.localization: # classify link 5 & 6
                    if data.iloc[i, 0] == labels_map[j]:
                        if j==0:
                            data_target.iat[i] = -1
                        elif j==1 or j==3:
                            data_target.iat[i] = 0
                        elif j==2 or j==4:
                            data_target.iat[i] = 1

                elif self.collision:  # classify intentional contact and collision
                    if data.iloc[i, 0] == labels_map[j]:
                        if j==0:
                            data_target.iat[i] = -1
                        elif j==1 or j==2:
                            data_target.iat[i] = 0
                        elif j==3 or j==4:
                            data_target.iat[i] = 1

                elif self.num_classes == 3 or self.collision: # classify noncontact & intentional contact and collision(almost same with the elif self.collision)
                    if data.iloc[i, 0] == labels_map[j]:
                        if j==0:
                            data_target.iat[i] = 0 # iat is faster than iloc when dealing with single element
                        elif j==1 or j==2:
                            data_target.iat[i] = 1
                        elif j==3 or j==4:
                            data_target.iat[i] = 2

                elif self.num_classes == 5: # classify all 5 type
                    if data.iloc[i, 0] == labels_map[j]:
                        data_target.iat[i] = j

                elif self.num_classes ==2: # classify contact or not
                    if data.iloc[i, 0] == labels_map[j]:
                        if j==0:
                            data_target.iat[i] = 0
                        else:
                            data_target.iat[i] = 1

                else: 
                    print('ERROR! num_classes should be 2 or 3 or 5')
                    exit()

        # print(data_input)
        # print(data_target)

        # why remove those with no contact? ? ?
        if self.localization or self.collision:
            data_input = data_input[data_target.iloc[:]!=-1]
            data_target = data_target[data_target.iloc[:]!=-1]
        # print(data_input)
        # print(data_target)

        self.data_input = data_input.reset_index(drop=True)
        self.data_target = data_target.reset_index(drop=True)
        # print(data_input)
        # print(data_target)

    def data_in_seq(self):

        dof = self.dof

        # resorting item position
        data = np.array( range(0, self.num_features_dataset * self.data_seq ))
        data = data.reshape(self.data_seq, self.num_features_dataset)

        joint_data_pos = []
        for j in range(dof):
            # (4,28) : [tau(t), tau_ext(t), e(t), de(t)]j
            if self.num_features == 4:
                column_index = [j, j+dof, j+dof*2, j+dof*3 ]
            elif self.num_features == 2:
                column_index = [j+dof*2, j+dof*3 ]
            
            elif self.num_features == 3:
                column_index = [j+dof, j+dof*2, j+dof*3 ]
                 
                
            row_index= range(self.data_seq-self.desired_seq, self.data_seq)
            join_data_matrix = data[:, column_index]

            joint_data_pos.append(join_data_matrix.reshape((len(column_index)*len(row_index))))
        
        joint_data_pos = np.hstack(joint_data_pos)

        # resorting (28,28)---> (4,28)(4,28)(4,28)(4,28)(4,28)(4,28)(4,28)

        self.data_input.columns = range(self.num_features_dataset * self.data_seq)
        self.data_input = self.data_input.loc[:][joint_data_pos]

class create_tensor_dataset(Dataset):
    
    def __int__(self, path="", num_classes=3, num_features=10,):
        self.path = path
        self.num_classes = num_classes
        self.num_features = num_features

        # tau_J,tau_J_d,tau_ext,q,q_d,dq,dq_d,e,de,etau_J
        # label,block_id,touch_type

        self.read_dataset()

    def read_dataset(self):
        data = pd.read_csv(self.path)
        data_input = data.iloc[:, 1:data.shape[1]]
        data_target = data['label']

if __name__ == '__main__':
    #data = pd.read_csv('../contactInterpretation-main/dataset/realData/contact_detection_train.csv')
    a = create_tensor_dataset_without_torque(collision=True)
    dataS, tar = a[703]
    print(dataS.size())
    print(tar)
    print(len(a))