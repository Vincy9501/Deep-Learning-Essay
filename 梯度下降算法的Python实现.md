# 1. 代数法

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

# 2. 矩阵法
```python
import numpy as np  
import matplotlib.pyplot as plt  
  
# 创建数据  
X = 2 * np.random.rand(100, 1)  
y = 4 + 3 * X + np.random.randn(100, 1)  
  
# plot() 函数是绘制二维图形的最基本函数  
# plot([x], y, [fmt], *, data=None, **kwargs)  
# fmt表示定义基本格式  
plt.plot(X, y, 'b.')  
# xlabel() 方法提供了 loc 参数来设置 x 轴显示的位置，可以设置为: 'left', 'right', 和 'center'， 默认值为 'center'。  
plt.xlabel("$x$", fontsize=18)  
plt.ylabel("$y$", rotation=0, fontsize=18)  
plt.axis([0, 2, 0, 15])  
  
# 线性回归  
  
# np.ones() 返回一个给定形状和类型的新数组，全是1，100行，1列  
# np.c_() 按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等  
# X.T.dot(X) 返回X^T 和 X 的点积  
# 这一句是给每个X添加偏差单位  
X_b = np.c_[np.ones((100, 1)), X]  
  
# np.linalg.inv 计算矩阵的逆矩阵  
# 这个式子直接去找\frac {\partial {L(b)}} {\partial {w}} = 0 时w的表达式  
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  
  
  
# 成本函数  
def cal_cost(theta, X, y):  
    m = len(y)  
    predictions = X.dot(theta)  # y = wX  
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))  
    return cost  
  
  
# 梯度下降函数  
def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):  
    m = len(y)  
    cost_history = np.zeros(iterations)  
    theta_history = np.zeros((iterations, 2))  
    for it in range(iterations):  
        prediction = np.dot(X, theta)  
        # w = w - 1/m * \alpha * X^T * (Xw - y)  
        theta = theta - (1 / m) * learning_rate * (X.T.dot((prediction - y)))  
        theta_history[it, :] = theta.T  
        cost_history[it] = cal_cost(theta, X, y)  
  
    return theta, cost_history, theta_history  
  
  
lr = 0.01  
n_iter = 1000  
  
theta = np.random.randn(2, 1)  
  
X_b = np.c_[np.ones((len(X), 1)), X]  
theta, cost_history, theta_history = gradient_descent(X_b, y, theta, lr, n_iter)  
  
print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0], theta[1][0]))  
print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))  
  
fig, ax = plt.subplots(figsize=(12, 8))  
  
ax.set_ylabel('J(Theta)')  
ax.set_xlabel('Iterations')  
_ = ax.plot(range(n_iter), cost_history, 'b.')  
  
  
def plot_GD(n_iter, lr, ax, ax1=None):  
    """  
    n_iter = no of iterations    lr = Learning Rate    ax = Axis to plot the Gradient Descent    ax1 = Axis to plot cost_history vs Iterations plot  
    """    _ = ax.plot(X, y, 'b.')  
    theta = np.random.randn(2, 1)  
  
    tr = 0.1  
    cost_history = np.zeros(n_iter)  
    for i in range(n_iter):  
        pred_prev = X_b.dot(theta)  
        theta, h, _ = gradient_descent(X_b, y, theta, lr, 1)  
        pred = X_b.dot(theta)  
  
        cost_history[i] = h[0]  
  
        if ((i % 25 == 0)):  
            _ = ax.plot(X, pred, 'r-', alpha=tr)  
            if tr < 0.8:  
                tr = tr + 0.2  
    if not ax1 == None:  
        _ = ax1.plot(range(n_iter), cost_history, 'b.')  
  
  
fig = plt.figure(figsize=(30, 25), dpi=200)  
fig.subplots_adjust(hspace=0.4, wspace=0.4)  
  
it_lr = [(2000, 0.001), (500, 0.01), (200, 0.05), (100, 0.1)]  
count = 0  
for n_iter, lr in it_lr:  
    count += 1  
  
    ax = fig.add_subplot(4, 2, count)  
    count += 1  
  
    ax1 = fig.add_subplot(4, 2, count)  
  
    ax.set_title("lr:{}".format(lr))  
    ax1.set_title("Iterations:{}".format(n_iter))  
    plot_GD(n_iter, lr, ax, ax1)  
  
plt.show()
```

![[Figure_3.png]]