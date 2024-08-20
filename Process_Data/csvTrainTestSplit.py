import pandas as pd
from sklearn.model_selection import train_test_split
#510个数据的时候是408+102
# 读取原始CSV文件
df = pd.read_csv('DATA/tactile_dataset_block.csv')

# 根据 block_id 进行分组
grouped = df.groupby('block_id')

# 将分组后的数据块列表化
blocks = [group for _, group in grouped]

# 使用 train_test_split 按比例划分训练集和测试集
train_blocks, test_blocks = train_test_split(blocks, test_size=0.2, random_state=42)

# 合并各自的分组数据
train_df = pd.concat(train_blocks).reset_index(drop=True)
test_df = pd.concat(test_blocks).reset_index(drop=True)

# 保存为新的CSV文件
train_df.to_csv('DATA/tactile_dataset_block_train.csv', index=False)
test_df.to_csv('DATA/tactile_dataset_block_test.csv', index=False)
