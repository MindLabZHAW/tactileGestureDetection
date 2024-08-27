import numpy as np
 
# 新建一个二维列表
t1 = np.array([[1, 2, 3], [4, 3, 4]])
print(t1.shape)
 
# 新建一个一维度列表,并更改数值
t2 = np.arange(24).reshape(6, 4)
print(t2.shape)
print(t2)
 
# 改为三维数组(2, 3, 4):2个模块,3行,每行4列
t3 = np.arange(24).reshape((2, 3, 4))
print(t3.shape)
print(t3)
 
# 将不确定维度的数组变为,一维度,方法1
print(t3.shape[0])
t3 = t3.reshape((t3.shape[0] * t3.shape[1] * t3.shape[2]))
print(t3)
# 将不确定维度的数组变为,一维度,方法2
t4 = np.arange(10).reshape((2, 5))
print(t4)
t4 = t4.flatten()
print(t4)
# 数组加常数,会在所有元素中加上数
t5 = np.arange(10).reshape((2, 5))
t5 = t5 + 1
print(t5)
 
# nan格式数据,not a number,inf格式数据,表示infinity,无穷的意思
t6 = np.arange(10).reshape((2, 5))
t6 = t6 / 0
print(t6)
 
#!!!!数组的加减乘除,为对应数上加减乘除, 维度不同时候,会广播计算
#!!!!! ,只要在任何情况下,有形状相同,都可以进行计算!!!!!!,