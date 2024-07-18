import json
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_catagorical

data_dir = "tactileGestureDetection/data"

image_data = []
labels = []
print(os.listdir(data_dir))

for subdir in os.listdir(data_dir):
    print(f"subdir is {subdir}")
    subdir_path = os.path.join(data_dir, subdir)
    print(f"subdir_path is {subdir_path}")

    if os.path.isdir(subdir_path):
        label = subdir.split("-")[1][1:]
    print(f"label is {label}")

    # the plot path
    plot_path = os.path.join(subdir_path,"plot")

    if os.path.exists(plot_path):
        for plot_file in  os.listdir(plot_path):
            if plot_file.endswith(".png"):
                plot_filepath = os.path.join(plot_path,plot_file)

                image = Image.open(plot_filepath).convert("L")
                image = image.resize((128,128))

                image_data.append(np.array(image))
                labels.append(label)


image_data = np.array(image_data)      
image_data = image_data/255.0

label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels)
one_hot_labels = to_catagorical(integer_labels)

np.save("label_encoder_classes.npy",label_encoder.classes_)

print(f"Processsed {len(image_data)} images")




