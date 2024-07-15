import json
import numpy as np


with open("tactileGestureDetection/data/0712-7FDT-1/20240712_154746_all_attributes.json","r") as file:
    data = json.load(file)



# preproccessing
timestamp = [item["timestamp"] for item in data]

attribute_name = np.unique([item["attribute_name"] for item in data])

print(attribute_name)


for item in data:
    print(item)