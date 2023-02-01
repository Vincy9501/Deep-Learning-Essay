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
