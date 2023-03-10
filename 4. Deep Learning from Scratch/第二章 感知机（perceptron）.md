
# 1. 感知机是什么

感知机接收多个输入信号，输出一个信号。但感知机的信号只有流/不流（1/0）两种取值，0-不传递信号，1-传递信号。

![[Pasted image 20230210154257.png]]
x1、x2是输入信号， y是输出信号，w1、w2是权重，○称为“神经元”或者“节点”。输入信号被送往神经元时，会被分别乘以固定的权重，神经元会计算传送过来的信号的总和，当总和超过**阈值**时，才会输出1。

因此用公式表达：

$$ y=\left\{
\begin{array}{rcl}
0       &      & {(w_1x_1 + w_2x_2)      \le     \theta}\\
1     &      & {(w_1x_1 + w_2x_2)      \gt     \theta}\\
\end{array} \right. $$

# 2. 简单逻辑电路

- 与门（AND gate）：仅在两个输入均为1时输出1，其他时候则输出0。
- 与非门（NAND gate）：仅在两个输入均为1时输出0，其他时候则输出1。
- 或门：只要有一个输入信号是1，输 出就为1。

# 3. 实现

## 3.1 简单实现

![[Pasted image 20230210155410.png]]

## 3.2 导入权重和偏置之后的实现

$$ y=\left\{
\begin{array}{rcl}
0       &      & {(w_1x_1 + w_2x_2 + b)      \le     0}\\
1     &      & {(w_1x_1 + w_2x_2 + b)      \gt     0}\\
\end{array} \right. $$

这里b叫做偏置，其实和之前的公式差不多，就是把$\theta$换成了b。

w1和w2是控制**输入信号的重要性**的参数，而**偏置**是**调整神经元被激活的容易程度**（输出信号为1的程度）的参数。

![[Pasted image 20230210160534.png]]

# 4. 局限性

## 4.1 异或门（XOR gate）

仅当x1或x2中的一方为1时，才会输出1。

但是感知机无法实现，为什么呢？

或门的情况下，当权重参数 (b, w1, w2) = (−0.5, 1.0, 1.0)时，感知机可用以下的式子表示：
$$ y=\left\{
\begin{array}{rcl}
0       &      & {(x_1 + x_2 - 0.5)      \le     0}\\
1     &      & {(x_1 + x_2 - 0.5)      \gt     0}\\
\end{array} \right. $$

![[Pasted image 20230210160802.png]]

或门在(x1, x2) = (0, 0)时输出0，在(x1, x2)为(0, 1)、(1, 0)、(1, 1)时输出1。

如果是异或门：

![[Pasted image 20230210160901.png]]

显然用一条直线无法把○和△分开。

## 4.2 线性和非线性

线性分不开，非线性就可以分开了。
由图所示这样的曲线分割而成的空间称为 **非线性空间**，由直线分割而成的空间称为**线性空间**。

![[Pasted image 20230210161107.png]]

# 5. 多层感知机

感知机可以“叠加层”表示异或门，先思考一下异或门的问题：

![[Pasted image 20230210163635.png]]

把与、与非、或门代入，即可实现异或门：

![[Pasted image 20230210163702.png]]

代码实现：

![[Pasted image 20230210163850.png]]

如果用感知机的方法，则如图所示：

![[Pasted image 20230210163924.png]]

异或门是一种多层结构的神经网络。这里，将最左边的一列称为第0层，中间的一列称为第1层，最右边的一列称为第2层。

实际上，与门、或门是单层感知机，而异或门是2层感知机。叠加了多层的感知机也称为**多层感知机（multi-layered perceptron）**。

多层感知机的工作流程是：
1. 第0层的两个神经元接收输入信号，并将信号发送至第1层的神经元。
2. 第1层的神经元将信号发送至第2层的神经元，第2层的神经元输出y。

## 6. 从与非门到计算机

使用感知机甚至可以表示计算机，因为计算机和感知机一样有输入和输出，会按照某个既定规则进行计算。理论上2层感知机就能构建计算机。

# 7. 总结

- 感知机是具有输入和输出的算法。给定一个输入后，将输出一个既定的值。
- 感知机将权重和偏置设定为参数。
- 使用感知机可以表示与门和或门等逻辑电路。
- 异或门无法通过单层感知机来表示。
- 使用2层感知机可以表示异或门。
- 单层感知机只能表示线性空间，而多层感知机可以表示非线性空间。
- 多层感知机（在理论上）可以表示计算机。
