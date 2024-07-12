import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_attribute(csv_file_path, plot_file_path, attributes, num_columns):
    # Ensure the plot_file_path directory exists; create if not
    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path)


    #generate a list of colors using a colormap
    cmap =  plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(num_columns)]

    # Iterate through each CSV file in csv_file_path directory
    for filename in os.listdir(csv_file_path):
        if filename.endswith(".csv"):
            # Check if the file starts with any attribute in the list
            if any(filename.startswith(attr) for attr in attributes):
                csv_file = os.path.join(csv_file_path, filename)
                plot_file = os.path.join(plot_file_path, os.path.splitext(filename)[0] + ".png")

                # Check if the plot file exists and delete it if it does
                if os.path.exists(plot_file):
                    os.remove(plot_file)
                    print(f"Existing plot {plot_file} deleted.")

                try:
                    # Read CSV file to handle files 
                    df = pd.read_csv(csv_file, sep=",")
                    #print(df.head())

                    # Print the number of columns in the CSV file
                    # print(f"File: {filename}, Number of columns: {len(df.columns)}")

                    # Extract timestamp and values for each column
                    timestamp = pd.to_datetime(df["timestamp"])
                    #print(f"timestamp is {timestamp}")
                    columns_to_plot = df.columns[1:num_columns+1]  # Adjust indices based on actual CSV structure

                    # Plotting
                    plt.figure(figsize=(12, 6))

                    for i, col in enumerate(columns_to_plot):
                        plt.plot(timestamp, df[col], color=colors[i], label=f'Column {i+1}')

                    plt.xlabel('Timestamp')
                    plt.ylabel('Values')
                    plt.title(f'Waveform of {filename}')
                    plt.legend(loc="center left",bbox_to_anchor = (1,0.5))
                    plt.grid(True)
                    plt.tight_layout()

                    # Save the plot
                    plt.savefig(plot_file)
                    plt.close()
                    print(f"Plot saved to {plot_file}")
                except Exception as e:
                    print(f"Error plotting {csv_file}: {e}")
def plot_2d_attribute(csv_file_path, plot_file_path):
    attributes_2 = ["elbow", "	elbow_d", "elbow_c", "delbow_c", "ddelbow_c"]
    plot_attribute(csv_file_path, plot_file_path, attributes_2, 2)


def plot_3d_attribute(csv_file_path, plot_file_path):
    attributes_3 = ["F_x_Cee", "F_x_Cload", "F_x_Ctotal"]
    plot_attribute(csv_file_path, plot_file_path, attributes_3, 3)


def plot_6d_attribute(csv_file_path, plot_file_path):
    attributes_6 = ["cartesian_contact", "cartesian_collision", "O_F_ext_hat_K ", "K_F_ext_hat_K", "O_dP_EE_d", "O_dP_EE_c", "O_ddP_EE_c "]
    plot_attribute(csv_file_path, plot_file_path, attributes_6, 6)

def plot_7d_attribute(csv_file_path, plot_file_path):
    attributes_7 = ["q", "q_d", "tau_ext_hat_filtered","joint_collision","tau_J", "tau_J_d", "theta", "dtheta","dtau_J","dq","dq_d","ddq_d","joint_contact"]
    plot_attribute(csv_file_path, plot_file_path, attributes_7, 7)
   

def plot_9d_attribute(csv_file_path, plot_file_path):
    attributes_9 = ["I_ee", "I_load", "I_total"]
    plot_attribute(csv_file_path, plot_file_path, attributes_9, 9)


def plot_16d_attribute(csv_file_path, plot_file_path):
    attributes_16 = ["O_T_EE", "O_T_EE_d", "F_T_EE", "F_T_NE", "NE_T_EE", "EE_T_K", "O_T_EE_c"]
    plot_attribute(csv_file_path, plot_file_path, attributes_16, 16)


# Example usage:
if __name__ == '__main__':
    csv_file_path = "data/0712"
    plot_file_path = "data/0712/plot"
    plot_2d_attribute(csv_file_path, plot_file_path)
    plot_3d_attribute(csv_file_path, plot_file_path)
    plot_7d_attribute(csv_file_path, plot_file_path)
    plot_6d_attribute(csv_file_path, plot_file_path)
    plot_9d_attribute(csv_file_path, plot_file_path)
    plot_16d_attribute(csv_file_path, plot_file_path)
