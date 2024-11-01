import os
import pandas as pd
import numpy as np

# STEP1: class used to label 0(No Contact) and 1(Gesture Name)
class make_folder_dataset:
    def __init__(self, gesture_name:str, folder_path:str,save_path) -> None:
        self.gesture_name = gesture_name
        
        self.path = folder_path
        self.save_path = save_path
        #os.makedirs(self.save_path)
        self.num_lines_per_message = 130
        self.df = pd.DataFrame()
        self.df_dataset = pd.DataFrame()
        self.tau = ['tau_J0','tau_J1', 'tau_J2', 'tau_J3', 'tau_J4', 'tau_J5', 'tau_J6']
        self.tau_d = ['tau_J_d0','tau_J_d1', 'tau_J_d2', 'tau_J_d3', 'tau_J_d4', 'tau_J_d5', 'tau_J_d6']
        self.tau_ext =['tau_ext0','tau_ext1','tau_ext2','tau_ext3','tau_ext4','tau_ext5','tau_ext6']

        self.q = ['q0','q1','q2','q3','q4','q5','q6']
        self.q_d = ['q_d0','q_d1','q_d2','q_d3','q_d4','q_d5','q_d6']

        self.dq = ['dq0','dq1','dq2','dq3','dq4','dq5','dq6']
        self.dq_d = ['dq_d0','dq_d1','dq_d2','dq_d3','dq_d4','dq_d5','dq_d6']


        self.e = ['e0','e1','e2','e3','e4','e5','e6']
        self.de = ['de0','de1','de2','de3','de4','de5','de6']
        self.etau = ['etau_J0','etau_J1', 'etau_J2', 'etau_J3', 'etau_J4', 'etau_J5', 'etau_J6']
    
    def _extract_array(self, data_dict:dict, data_frame:str, header:list, n:int):
            dof = 7
            x, y = data_frame[n].split(':')
            y = y.replace('[','')
            y = y.replace(']','')
            y = y.replace('\n','')

            y = y.split(',')
            for i in range(dof):
                data_dict[header[i]].append(float(y[i]))

    def extract_robot_data(self):
        # it extracts robot data from all_data.txt
        f = open(self.path +'/'+ 'all_data.txt', 'r')
        lines = f.readlines()

        keywords = ['time'] + self.tau + self.tau_d + self.tau_ext + self.q + self.q_d + self.dq + self.dq_d 

        data_dict = dict.fromkeys(keywords)
        for i in keywords:
            data_dict[i]=[0]
        
        for i in range(int(len(lines)/self.num_lines_per_message)):
            data_frame = lines[i*self.num_lines_per_message:(i+1)*self.num_lines_per_message]
            
            x, y = data_frame[3].split(':')
            time_ = int(y)-int(int(y)/1000000)*1000000

            x, y = data_frame[4].split(':')
            time_ = time_+int(y)/np.power(10,9)

            data_dict['time'].append(time_)
            
            self._extract_array(data_dict,data_frame,self.tau, 25)
            self._extract_array(data_dict,data_frame,self.tau_d, 26)
            self._extract_array(data_dict,data_frame, self.tau_ext, 37)
            
            self._extract_array(data_dict,data_frame,self.q, 28)
            
            self._extract_array(data_dict,data_frame, self.q_d, 29)
            self._extract_array(data_dict,data_frame, self.dq, 30)
            self._extract_array(data_dict,data_frame, self.dq_d, 31)
        
       
        self.df = pd.DataFrame.from_dict(data_dict)
        self.df = self.df.drop(index=0).reset_index()
        
        for i in range(len(self.e)):
            self.df[self.e[i]] = self.df[self.q_d[i]]-self.df[self.q[i]]
        for i in range(len(self.de)):
            self.df[self.de[i]] = self.df[self.dq_d[i]]-self.df[self.dq[i]]
        for i in range(len(self.etau)):
            self.df[self.etau[i]] = self.df[self.tau_d[i]]-self.df[self.tau[i]]

        # Print the DataFrame after extraction to verify content
        # print("DataFrame after extraction:")
        # print(self.df.head())

        #self.df.to_csv(self.save_path +'robot_data.csv',index=False)

    def get_labels(self):
        # Load the gesture labeled (contact- noncontact) data
        gesture_label_path = os.path.join(self.path, 'gesture_label.csv')
        
        # Initialize DataFrame for labels
        self.df['label_idx'] = 0
        self.df['label_name'] = 'NC'
        
        # Check if gesture_label.csv exists and is not empty
        if not os.path.exists(gesture_label_path) or os.stat(gesture_label_path).st_size == 0:
            print(f"Warning: Empty or missing gesture_label.csv in {self.path}. All labels set to 0 and 'NC'.")
        else:
            gesture_label = pd.read_csv(gesture_label_path)
            
            # Check if gesture_label is empty after loading
            if gesture_label.empty:
                print(f"Warning: gesture_label.csv is empty. All labels set to 0 and 'NC'.")
            else:
                # Adjust time column in gesture_label
                gesture_label['time'] = gesture_label['time_sec'] + gesture_label['time_nsec'] - self.df['time'][0]
                
                # Compute the time difference and determine gesture events
                time_dev = gesture_label['time'].diff()
                gesture_events_index = np.append([0], gesture_label['time'][time_dev > 0.05].index.values)
                gesture_events_index = np.append(gesture_events_index, gesture_label['time'].shape[0] - 1)

                # Check if gesture_events_index is empty
                if len(gesture_events_index) == 0:
                    print(f"Warning: No gesture events found in {gesture_label_path}. All labels set to 0 and 'NC'.")
                else:
                    self.df['time'] = self.df['time'] - self.df['time'][0]
                    gesture_count = 0
                    self.df['label'] = 0

                    for i in range(self.df['time'].shape[0]):
                        if gesture_count >= len(gesture_events_index):
                            break
                        if (self.df['time'][i] - gesture_label['time'][gesture_events_index[gesture_count]]) > 0:
                            gesture_count += 1
                            if gesture_count >= len(gesture_events_index):
                                break
                            for j in range(i, self.df['time'].shape[0]):
                                self.df.loc[j, 'label_idx'] = 1
                                self.df.loc[j, 'label_name'] = self.gesture_name
                                if gesture_count >= len(gesture_events_index) or \
                                (self.df['time'][j] - gesture_label['time'][gesture_events_index[gesture_count] - 1]) > 0:
                                    i = j
                                    break

        # Print the DataFrame before saving to verify content
        # print("DataFrame before saving:")
        # print(self.df.head())
        
        # Save the labeled data
        output_file = os.path.join(self.save_path, '1_labeled_data.csv')
        self.df.to_csv(output_file, index=False)
        print(f"1_labeled_data.csv saved to {output_file}")

# function used to run STEP1
def labelData(gesture_name, raw_data_dir, data_save_dir):
    os.makedirs(data_save_dir,exist_ok= True)
    instance = make_folder_dataset(gesture_name, raw_data_dir, data_save_dir)
    instance.extract_robot_data()
    instance.get_labels()
    return instance.df


# STEP2：Use sliding window to generate training and testing data
def windowData(label_data, data_save_dir, window_size=28, step_size=14):
    # 初始化存储窗口信息的列表
    windowed_data = []

    # 初始化 window_id
    window_id = 0

    # 循环以 step_size 步长遍历数据
    for start in range(0, len(label_data) - window_size + 1, step_size):
        end = start + window_size

        window = label_data.iloc[start:end].copy()

        # 生成 window_id
        window['window_id'] = window_id
        window_id += 1  # 每次迭代增加 window_id
        
        # 判断窗口内 touch_type 是否全为 'NC'
        unique_label_name = window['label_name'].unique()
        if len(unique_label_name) == 1 and unique_label_name[0] == 'NC':
            window_label_name = 'NC'
            window_label_idx = 0
        else:
            # 如果不是全为 'NC'，取窗口内唯一的非 'NC' 的 label_name
            non_nc_label_name = [t for t in unique_label_name if t != 'NC']
            window_label_name = non_nc_label_name[0] if non_nc_label_name else 'NC'
            window_label_idx = 1 if non_nc_label_name else 0
        
        # 给窗口内所有行赋值 window_gesture_idx, window_gesture_name
        window['window_gesture_idx'] = window_label_idx
        window['window_gesture_name'] = window_label_name
        
        # 添加窗口数据到列表
        windowed_data.append(window)

    # 合并所有窗口数据
    windowed_df = pd.concat(windowed_data, ignore_index=True)

    
    data_save_name =  data_save_dir + '2_labeled_window_dataset.csv'
    if os.path.exists(data_save_name):
        os.remove(data_save_name)

    # 保存到新的CSV文件
    windowed_df.to_csv(data_save_name, index=False)
        

    print(f"{data_save_name} 文件已生成，总行数 = {len(windowed_df)}")

    return windowed_df