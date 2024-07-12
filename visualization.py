import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_7d_attribute(csv_file_path, plot_file_path):
    # Ensure the plot_file_path directory exists; create if not
    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path)

    # Attributes to plot (with 7 dimentions)
    attributes = [
        "q", "q_d", "tau_ext_hat_filtered", "tau_J", "tau_J_d", "theta", "dtheta"
    ]
    
    # Define colors for each attribute
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Iterate through each CSV file in csv_file_path directory
    for filename in os.listdir(csv_file_path):
        if filename.endswith(".csv"):
            # Check if the file starts with any attribute in the list
            if any(filename.startswith(attr) for attr in attributes):
                csv_file = os.path.join(csv_file_path, filename)
                plot_file = os.path.join(plot_file_path, os.path.splitext(filename)[0] + ".png")

                #check if the plot file exists and delete it if it does
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
                    columns_to_plot = df.columns[1:8]  # Adjust indices based on actual CSV structure

                    # Plotting
                    plt.figure(figsize=(12, 6))

                    for i, col in enumerate(columns_to_plot):
                        plt.plot(timestamp, df[col], color=colors[i], label=f'Column {i+1}')

                    plt.xlabel('Timestamp')
                    plt.ylabel('Values')
                    plt.title(f'Waveform of {filename}')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()

                    # Save the plot
                    plt.savefig(plot_file)
                    plt.close()
                    print(f"Plot saved to {plot_file}")
                except Exception as e:
                    print(f"Error plotting {csv_file}: {e}")

# Example usage:
if __name__ == '__main__':
    csv_folder_path = "data/0712"
    plot_folder_path = "data/0712/plot"
    plot_7d_attribute(csv_folder_path, plot_folder_path)
