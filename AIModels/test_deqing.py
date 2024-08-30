import pandas as pd

blocked_data = pd.read_csv("DATA/tactile_dataset_block_train.csv")
grouped = blocked_data.groupby('block_id')
for blockid, block in grouped:
    print('block_id is ',blockid,'lenth is ', block.shape[0])