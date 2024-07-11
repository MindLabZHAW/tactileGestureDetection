import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

csv_file_path = "data/0711/20240711_152501.csv"

data_preview = pd.read_csv(csv_file_path, nrows=5)
print(data_preview.head())
#csv_file_path = sys.argv[1]
data = pd.read_csv(csv_file_path,header = 0)

#print(data.head())


data["timestamp"] = pd.to_datetime(data["timestamp"])

attributes = data["attribute_name"].unique()

print(f"attributes in data: {attributes}")


# Create a subfolder for plots
# plot_dir = os.path.splitext(csv_file_path)[0]
# if not os.path.exists(plot_dir):
#     os.makedirs(plot_dir)


# function to plot waveform for a given attribute
def plot_waveform(attribute_name):
    attribute_data = data[data["attribute_name"] == attribute_name]

    if attribute_data.empty:
        print(f"No data found for attribute: {attribute_name}")
        return
    
    alue_columns = attribute_data.columns.difference(['timestamp', 'attribute_name'])

    for col in value_columns:
        plt.plot(attribute_data["timestamp"],attribute_data[col])
    plt.title(f"Waveform of {attribute_name}")
    plt.xlabel("Timestamp")
    plt.ylabel("Values")
    plt.show()

    # #save the plot
    # plot_path = os.path.join(plot_dir,f"{attribute_name}_waveform.png")
    # #plt.savefig(plot_path)
    # plt.close()



for attribute in attributes:
    plot_waveform(attribute)