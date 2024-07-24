import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class make_folder_dataset:
    def __init__(self, folder_path: str, plot_save_path: str) -> None:
        self.path = folder_path
        self.plot_save_path = plot_save_path
        self.num_lines_per_message = 130
        self.df = pd.DataFrame()
        self.df_dataset = pd.DataFrame()
        self.tau = ['tau_J0', 'tau_J1', 'tau_J2', 'tau_J3', 'tau_J4', 'tau_J5', 'tau_J6']
        self.tau_d = ['tau_J_d0', 'tau_J_d1', 'tau_J_d2', 'tau_J_d3', 'tau_J_d4', 'tau_J_d5', 'tau_J_d6']
        self.tau_ext = ['tau_ext0', 'tau_ext1', 'tau_ext2', 'tau_ext3', 'tau_ext4', 'tau_ext5', 'tau_ext6']
        self.q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
        self.q_d = ['q_d0', 'q_d1', 'q_d2', 'q_d3', 'q_d4', 'q_d5', 'q_d6']
        self.dq = ['dq0', 'dq1', 'dq2', 'dq3', 'dq4', 'dq5', 'dq6']
        self.dq_d = ['dq_d0', 'dq_d1', 'dq_d2', 'dq_d3', 'dq_d4', 'dq_d5', 'dq_d6']
        self.e = ['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']
        self.de = ['de0', 'de1', 'de2', 'de3', 'de4', 'de5', 'de6']
        self.etau = ['etau_J0', 'etau_J1', 'etau_J2', 'etau_J3', 'etau_J4', 'etau_J5', 'etau_J6']
        os.makedirs(self.plot_save_path, exist_ok=True)

    def _extract_array(self, data_dict: dict, data_frame: str, header: list, n: int):
        dof = 7
        x, y = data_frame[n].split(':')
        y = y.replace('[', '').replace(']', '').replace('\n', '').split(',')
        for i in range(dof):
            data_dict[header[i]].append(float(y[i]))

    def extract_robot_data(self):
        f = open(os.path.join(self.path, 'all_data.txt'), 'r')
        lines = f.readlines()
        keywords = ['time'] + self.tau + self.tau_d + self.tau_ext + self.q + self.q_d + self.dq + self.dq_d
        data_dict = dict.fromkeys(keywords)
        for i in keywords:
            data_dict[i] = [0]

        for i in range(int(len(lines) / self.num_lines_per_message)):
            data_frame = lines[i * self.num_lines_per_message:(i + 1) * self.num_lines_per_message]
            x, y = data_frame[3].split(':')
            time_ = int(y) - int(int(y) / 1000000) * 1000000
            x, y = data_frame[4].split(':')
            time_ = time_ + int(y) / np.power(10, 9)
            data_dict['time'].append(time_)
            self._extract_array(data_dict, data_frame, self.tau, 25)
            self._extract_array(data_dict, data_frame, self.tau_d, 26)
            self._extract_array(data_dict, data_frame, self.tau_ext, 37)
            self._extract_array(data_dict, data_frame, self.q, 28)
            self._extract_array(data_dict, data_frame, self.q_d, 29)
            self._extract_array(data_dict, data_frame, self.dq, 30)
            self._extract_array(data_dict, data_frame, self.dq_d, 31)

        self.df = pd.DataFrame.from_dict(data_dict)
        self.df = self.df.drop(index=0).reset_index(drop=True)
        for i in range(len(self.e)):
            self.df[self.e[i]] = self.df[self.q_d[i]] - self.df[self.q[i]]
        for i in range(len(self.de)):
            self.df[self.de[i]] = self.df[self.dq_d[i]] - self.df[self.dq[i]]
        for i in range(len(self.etau)):
            self.df[self.etau[i]] = self.df[self.tau_d[i]] - self.df[self.tau[i]]
        self.ros_time = self.df['time'][0]
        self.df['time'] = self.df['time'] - self.ros_time
     

    def plot_data(self, targetA, targetB):
        if targetA == self.q and targetB == self.q_d:
            plt.figure(figsize=(16,8))
            self.df.plot(x='time', y=targetA + targetB)
            plt.title('q and q_d combined plot')
            plt.xlabel('Time (sec)')
            plt.ylabel('Value')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            file_path = os.path.join(self.plot_save_path, 'q_q_d_combined.png')
            plt.savefig(file_path)
            plt.close()
            print(f'{file_path} generated successfully.')

        elif targetA == self.tau_ext and targetB == self.tau:
            plt.figure(figsize=(12,8))
            self.df.plot(x='time', y=targetA + targetB)
            plt.title('tau_ext and tau combined plot')
            plt.xlabel('Time (sec)')
            plt.ylabel('Value')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            file_path = os.path.join(self.plot_save_path, 'tau_ext_tau_combined.png')
            plt.savefig(file_path)
            plt.close()
            print(f'{file_path} generated successfully.')

            for i in range(len(targetA)):
                plt.figure(figsize=(24,8))
                self.df.plot(x='time', y=[targetA[i], targetB[i]])
                plt.title(f'{targetA[i]} and {targetB[i]} plot')
                plt.xlabel('Time (sec)')
                plt.ylabel('Value')
                plt.tight_layout()
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                file_path = os.path.join(self.plot_save_path, f'{targetA[i]}_{targetB[i]}.png')
                plt.savefig(file_path)
                plt.close()
                print(f'{targetA[i]}_{targetB[i]}.png generated successfully.')


def process_data_and_plot(dataset_folder_path, plot_save_path):
    # Clear the plot_save_path directory
    if os.path.exists(plot_save_path):
        for file_name in os.listdir(plot_save_path):
            file_path = os.path.join(plot_save_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(plot_save_path)

    file_path = dataset_folder_path
    plot_path = plot_save_path
    #os.makedirs(plot_path, exist_ok=True)
    data = make_folder_dataset(file_path, plot_path)
    data.extract_robot_data()
    targetA = data.q
    targetB = data.q_d
    data.plot_data(targetA, targetB)
    targetA =data.tau_ext
    targetB = data.tau
    data.plot_data(targetA, targetB)

dataset_path = '../tactileGestureDetection/DATA/0722-7RST-1'
plot_save_path = '../tactileGestureDetection/DATA/0722-7RST-1/plots/'

process_data_and_plot(dataset_path, plot_save_path)
