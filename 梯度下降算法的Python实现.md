```python
# 假设函数yhat = wx + b  
# 损失函数loss = (yhat - y)**2 / N  
# numpy用于创建x和y数据  
import numpy as np  
  
'''  
1. 初始化  
一开始并不知道w和b分别是什么，所以可以使用梯度下降去找出它们  
'''  
x = np.random.randn(10, 1)  # 随机生成遵循正态分布的数据  
y = 2 * x + np.random.rand()  # y将等于一些随机参数  
# 实际参数  
w = 0.0  
b = 0.0  
# 其他参数  
learning_rate = 0.01  # 步长  
'''  
2. 创建梯度下降函数  
BGD  
以梯度 * 步长的幅度下降  
'''  
def descend(x, y, w, b, learning_rate):  
    dldw = 0.0  # l对w的导数  
    dldb = 0.0  # l对b的导数  
    N = x.shape[0]  # 我们需要知道训练集中数据点的个数，在这里使用shape[0]返回数组的维度，x.shape[0]返回行数，也就是10  
    # loss = (y - (wx+b))**2    # zip() 将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表  
    for xi, yi in zip(x, y):  
        dldw += -2 * xi * (yi - (w * xi + b))  
        dldb += -2 * (yi - (w * xi + b))  
    # 对w更新  
    w = w - learning_rate * (1 / N) * dldw  
    # 对b更新  
    b = b - learning_rate * (1 / N) * dldb  
  
    return w, b  
'''  
3. 迭代更新  
'''  
for epoch in range(600):  
    w, b = descend(x, y, w, b, learning_rate)  
    yhat = w * x + b  
    loss = np.divide(np.sum((y - yhat) ** 2, axis=0), x.shape[0])  # axis 用于按行求和或者按列求和，axis = 0表示按列求和  
    print(f'{epoch} loss is {loss}, parameters w:{w}, b:{b}')
```