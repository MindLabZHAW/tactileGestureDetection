import pandas as pd
from sklearn.model_selection import train_test_split
#510个数据的时候是408+102
# 13522+3381(0910新数据window切割)
# 读取原始CSV文件
# df = pd.read_csv('DATA/labeled_window_dataset.csv')
df = pd.read_csv('DATA/labeled_window_dataset_Normalization.csv')

# 根据 block_id 进行分组
grouped = df.groupby('window_id')

# 将分组后的数据块列表化
blocks = [group for _, group in grouped]

# 使用 train_test_split 按比例划分训练集和测试集
train_blocks, test_blocks = train_test_split(blocks, test_size=0.2, random_state=42)

# 合并各自的分组数据
train_df = pd.concat(train_blocks).reset_index(drop=True)
print('Size of train_df is ', train_df.shape)
test_df = pd.concat(test_blocks).reset_index(drop=True)
print('Size of test_df is ', test_df.shape)

# 保存为新的CSV文件
""" train_df.to_csv('DATA/labeled_window_dataset_train.csv', index=False)
test_df.to_csv('DATA/labeled_window_dataset_test.csv', index=False) """

train_df.to_csv('DATA/labeled_window_dataset_train_Normalization.csv', index=False)
test_df.to_csv('DATA/labeled_window_dataset_test_Normalization.csv', index=False)
