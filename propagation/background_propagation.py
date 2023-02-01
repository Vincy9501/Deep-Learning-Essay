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

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        # np.dot 矩阵的乘法运算，返回两个数的点积
        self.z = np.dot(X, self.W1)
        # 激活函数，用sigmoid压缩
        self.z2 = self.sigmoid(self.z)
        # 对隐藏层和第二组权重进行另一个点积
        self.z3 = np.dot(self.z2, self.W2)
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
