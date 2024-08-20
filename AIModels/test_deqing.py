import pandas as pd
import numpy as np


df = pd.read_csv('DATA/tactile_dataset_block_train.csv')

grouped = df.groupby('block_id')

lenth = []
for block_id, group in grouped:
    lenth.append(group.shape[0])
print(len(lenth))

