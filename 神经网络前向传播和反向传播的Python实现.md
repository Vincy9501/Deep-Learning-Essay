
# 1. 引言

本质上，神经网络是由**突触**连接的**神经元**集合。该集合分为三个主要层：输入层、隐藏层和输出层。

在人工神经网络中，有多个输入，称为**特征**，它们至少产生一个输出——称为**标签**。

![[Pasted image 20230201153606.png]]

在上图中，圆圈代表神经元，而线代表突触。突触的作用是获取输入和权重并将二者相乘。

权重视为神经元之间连接的“强度”，权重主要定义神经网络的输出。

之后，应用**激活函数**以返回输出。（这里的激活函数，就是我们之前提到的[[神经网络的反向传播（back propagation）#^0c0932|sigmoid或者ReLU]]

下面是一个简单的前馈神经网络（feedforward）如何工作的简要概述：

1.  将输入作为一个矩阵（二维数组）；
2.  将输入乘以一组权重；
3.  应用激活函数；
4.  返回一个输出；
5.  通过计算模型的**期望输出**与**预测输出**之间的差异来计算误差。这是一个称为梯度下降的过程，我们可以用它来改变权重；
6.  然后根据在步骤 5 中发现的错误调整权重；
7.  为了训练，这个过程要重复 1,000 多次。训练的数据越多，我们的输出就越准确。

神经网络的核心很简单，**只是对输入和权重执行矩阵乘法，并应用激活函数**。当通过损失函数的梯度调整权重时，网络会适应变化以产生更准确的输出。

我们的神经网络将对具有三个输入和一个输出的单个隐藏层进行建模。在网络中，我们将根据前一天学习了多少小时和睡了多少小时的输入来预测考试分数。输出是“测试分数”。

示例数据：

![[Pasted image 20230201155506.png|325]]

# 1. 前向传播

```python
import numpy as np  
  
# X = (睡觉时间, 学习时间), y = 测试分数  
# dtype表示数据类型  
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)  
y = np.array(([92], [86], [89]), dtype=float)  
  
# 标准化单位，因为我们的输入以小时为单位，但我们的输出是 0-100 的测试分数  
X = X / np.amax(X, axis=0)  
y = y / 100  
  
  
# 我们的输入数据X是一个 3x2 矩阵, 输出数据y是一个 3x1 矩阵  
# 矩阵中的每个元素都X需要乘以相应的权重，然后与隐藏层中每个神经元的所有其他结果相加  
  
# 为了获得隐藏层的最终值，我们需要应用激活函数  
# 好处是输出在0-1内，更容易改变权重  
  
  
class NeuralNetwork(object):  
    # 每当创建类的对象时调用  
    # self代表一个类的实例，需要访问类中的任何变量或方法  
  
    def __init__(self):  
        self.inputSize = 2  
        self.outputSize = 1  
        self.hiddenSize = 3  
  
        # 权重  
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # (3x2) weight matrix from input to hidden layer  
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # (3x1) weight matrix from hidden to output layer  
  
    def forward(self, X):  
        # np.dot 矩阵的乘法运算，返回两个数的点积  
        self.z = np.dot(X, self.W1)  
        # 激活函数，用sigmoid压缩  
        self.z2 = self.sigmoid(self.z)  
        # 对隐藏层和第二组权重进行另一个点积  
        self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of 3x1 weights  
        # 压缩  
        o = self.sigmoid(self.z3)  
        return o  
  
    def sigmoid(self, s):  
        return 1 / (1 + np.exp(-s))  
  
  
NN = NeuralNetwork()  
  
o = NN.forward(X)  
  
print("预测输出：" + str(o))  
print("实际输出：" + str(y))
```

```python
预测输出：[[0.6489327 ]
 [0.66220512]
 [0.6966329 ]]
实际输出：[[0.92]
 [0.86]
 [0.89]]
```

# 2. 反向传播

```python
# 通过使用损失函数来计算网络与目标输出的距离  
# 为了弄清楚改变权重的方向，我们需要找到损失相对于权重的变化率  
# 也就是求导  
  
"""  
计算权重的变化方式：  
1. 通过取预测输出和实际输出 (y) 的差值来找到输出层的误差范围  
2. sigmoid 激活函数的导数应用于输出层误差  
3. 与我们的第二个权重矩阵执行点积，使用输出层误差的增量输出总和来计算隐藏层对输出误差的贡献程度  
4. 应用我们的 sigmoid 激活函数的导数来计算第二层的输出和  
5. 调整第一层的权重  
  
"""  
  
import numpy as np  
  
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)  
y = np.array(([92], [86], [89]), dtype=float)  
  
X = X / np.amax(X, axis=0)  
y = y / 100  
  
  
class NeuralNetwork(object):  
    def __init__(self):  
        self.inputSize = 2  
        self.outputSize = 1  
        self.hiddenSize = 3  
  
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # (3x2) weight matrix from input to hidden layer  
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # (3x1) weight matrix from hidden to output layer  
  
    def forward(self, X):  
        # np.dot 矩阵的乘法运算，返回两个数的点积  
        self.z = np.dot(X, self.W1)  
        # 激活函数，用sigmoid压缩  
        self.z2 = self.sigmoid(self.z)  
        # 对隐藏层和第二组权重进行另一个点积  
        self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of 3x1 weights  
        # 压缩  
        o = self.sigmoid(self.z3)  
        return o  
  
    def sigmoid(self, s):  
        return 1 / (1 + np.exp(-s))  
  
    # 反向传播部分  
  
    def sigmoidPrime(self, s):  
        # 激活函数求导  
        return s * (1 - s)  
  
    def backward(self, X, y, o):  
        # 预测输出与目标输出之间的差距  
        self.o_error = y - o  
        self.o_delta = self.o_error * self.sigmoidPrime(o)  
  
        self.z2_error = self.o_delta.dot(  
            self.W2.T)  
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)  
  
        # 更新权重w1和w2  
        self.W1 += X.T.dot(self.z2_delta)  
        self.W2 += self.z2.T.dot(self.o_delta)  
  
    def train(self, X, y):  
        o = self.forward(X)  
        self.backward(X, y, o)  
  
  
NN = NeuralNetwork()  
  
o = NN.forward(X)  
for i in range(1000):  
    print("输入：" + str(X))  
    print("实际输出：" + str(y))  
    print("预测输出：" + str(NN.forward(X)))  
    print("Loss：" + str(np.mean(np.square(y - NN.forward(X)))))  
    print("\n")  
    NN.train(X, y)
```

```python
输入：[[0.66666667 1.        ]
 [0.33333333 0.55555556]
 [1.         0.66666667]]
实际输出：[[0.92]
 [0.86]
 [0.89]]
预测输出：[[0.89832071]
 [0.86358545]
 [0.90859076]]
Loss：0.000276154424582096
```