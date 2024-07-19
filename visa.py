import json
import pandas as pd
import os
import plotly.express as px
import plotly.io as pio
import cufflinks as cf
from plotly.offline import iplot, init_notebook_mode

input_file_path="data/0719-7RST-1"
for filename in os.listdir(input_file_path):
    if filename.endswith(".json"):
        # Check if the file starts with any attribute in the list
        input_file = os.path.join(input_file_path, filename)
        #plot_file = os.path.join(plot_file_path, os.path.splitext(filename)[0] + ".png")    

        # Check if the plot file exists and delete it if it does
        # if os.path.exists(plot_file):
        #     os.remove(plot_file)
        #     print(f"Existing plot {plot_file} deleted.")    

        try:
             # read the JSON  file 
             with open(input_file,"r") as f:
                 data = json.load(f)

            # convert JSON data to DataFrame
             records = []
             for entry in data:
                timestamp = entry["timestamp"]
                attribute_name = entry["attribute_name"]
                for key,value in entry["values"].items():
                     records.append({
                            "timestamp": timestamp,
                            "attribute_name": attribute_name,
                            "variable": key,
                            "value": value
                        }
                    )
                     
             t = records.q
             print(t)
                 
                 #print(records)
        except Exception as e:
                        print(f"Failed to process file {input_file}: {e}")