
# 1. YOLO vs Faster R-CNN

1、统一网络：YOLO没有显示求取region proposal的过程。Faster R-CNN中尽管RPN与fast rcnn共享卷积层，但是在模型训练过程中，需要反复训练RPN网络和fast rcnn网络。YOLO只需要Look Once.

2、YOLO统一为一个回归问题，而Faster R-CNN将检测结果分为两部分求解：物体类别（分类问题）、物体位置即bounding box（回归问题）。

# 2. 版本演变

## 2.1 v1

- 核心思想：用整张图作为网络的输入，直接在输出层回归bounding box的位置和类别
- 实现方法：一幅图像分成S×S个网格（grid cell），如果某个object的中心落在这个网格中那么这个网格就负责预测这个object
- 网络结构：GoogLeNet网络构成：24个卷积层+2个全连接层，卷积层用的7×7卷积
- 损失函数：误差平方和（sum-squared error loss）
- 缺点：一个网格只能检测出一个object，因为IoU只取最大的；损失函数导致对小目标的检测不好。

问题：怎么判断物体在网格中？

![[Pasted image 20230212201455.png]]

- 每个网格要预测B个bounding box，每个bounding box要回归自身位置和预测confidence值，这个值是用来代表预测的box中含有object的置信度和准确度的信息。
$$Pr(Object)*IOU^{truch}_{pred}$$

如果有object落在一个grid cell里，则第一项取1，否则取0。 第二项是预测的bounding box和实际的groundtruth之间的IoU值。

- 每个bounding box要预测(x, y, w, h)和confidence共5个值，每个网格还要预测一个类别信息，记为C类。每个网格要预测B个bounding box还要预测C个categories。输出就是S x S x (5*B+C)的一个tensor。

问：什么是IoU值呢？
答：它用来区分候选框是属于正样本还是负样本，计算公式是A和B的交并比。

问：格子的限制？
答：每个格子可以预测B个bounding box，但是最终只选择IOU最高的bounding box作为物体检测输出，即每个格子最多只预测出一个物体。

## 2.2 v2

- 改进方向：**Better**、**Faster**、**Stronger**
- 核心思想：将图片输入到darknet19网络中提取特征图，然后输出目标框类别信息和位置信息；提出联合训练算法，把两种数据集混合，用分层的观点分类物体
- 网络结构：Darknet-19
- 重要改进：passthrough操作，将生成的14x14x2048与原始的14x14x1024进行concat操作；引入anchor，调整位置预测为偏移量预测。
- 缺点：小目标检测结果有待提升。


- 改进点：
	- 批量归一化（Batch Normalization），解决反向传播过程中的梯度消失和梯度爆炸问题，降低对一些超参数的敏感性，并且每个batch分别进行归一化时起到正则化效果；
	- 高分辨率分类（High resolution classifier），就是224\*224图像进行分类模型预训练之后采用448\*448高分辨率样本训练；
	- 用锚框（先验框）卷积（Convolution with anchor boxes），方便微调边框位置；
	- 维度聚类（Dimension clusters），对训练集中标注的边框进行K-means聚类分析，以寻找尽可能匹配样本的边框尺寸。
	- 直接定位预测（Direct location prediction），
	- 细粒度特征（ **Fine-Grained Features**），引入passthrough层保留细节信息
	- 不同尺寸训练（Multi-ScaleTraining），每迭代几次都会改变网络参数。

## 2.3 v3

- 改进方向：缓解增加深度的梯度消失问题、多尺寸预测
- 核心思想：将图片输入到darknet53网络中提取特征图，借鉴特征金字塔网络思想，将高级和低级语义信息进行融合，在低、中、高三个层次上分别预测目标框，最后输出三个尺度的特征图信息（52×52×75、26×26×75、13×13×75）。
- 网络结构：Darknet-53
- 重要改进：引入残差；借鉴特征金字塔，在三个不同的尺寸上分别进行预测。

- 改进点：
	- feature map中的每一个cell都会预测3个边界框（bounding box） ，每个bounding box都会预测每个框的位置、一个objectness prediction、N个类别。
	- 损失函数：多标签分类。

## 2.4 v4

- 改进方向：生成新的网络、扩大感受野、引入PAN结构、Mish激活函数、Mosaic数据增强
- 核心思想：基本与v3类似
- 网络结构：CSPDarknet53（含spp）Neck: FPN+PAN
- 重要改进：将CSP结构融入Darknet53中，生成了新的主干网络CSPDarkent53；采用SPP空间金字塔池化来扩大感受野；在Neck部分引入PAN结构，即FPN+PAN的形式；引入Mish激活函数；引入Mosaic数据增强；训练时采用CIOU_loss ，同时预测时采用DIOU_nms


框架原理：

![[Pasted image 20230212214106.png]]

- CSP Darknet53
	- 借鉴了CSPNet，CSPNet全称是Cross Stage Partial Networks，也就是跨阶段局部网络。
	- 解决了其他大型卷积神经网络框架Backbone中网络优化的梯度信息重复问题。
	- 进入每个stage先将数据划分为两部分，两个分支的信息在交汇处进行Concat拼接。
- SPP结构
	- 解决不同尺寸的特征图如何进入全连接层。
	- 用到这里是为了增加感受野。
- PAN结构
	- 相比于原始的PAN结构，YOLOV4实际采用的PAN结构将addition的方式改为了concatenation。
	- 由于FPN结构是自顶向下的，将高级特征信息以上采样的方式向下传递，但是融合的信息依旧存在不足，因此YOLOv4在FPN之后又添加了PAN结构，再次将信息从底部传递到顶部。


训练策略：
- CutMax
	- 随机生成一个裁剪框Box,裁剪掉A图的相应位置，然后用B图片相应位置的ROI放到A图中被裁剪的区域形成新的样本，ground truth标签会根据patch的面积按比例进行调整，比如0.6像狗，0.4像猫，计算损失时同样采用加权求和的方式进行求解。


# 总结

- v1版本，是用整张图作为输入，把图分为很多个格子，object的中心落在哪个格子就用哪个格子去预测，每个格子预测很多个bounding box，每个bounding box回归自身位置并且提供confidence值（置信度和准确度）。
- v2版本总体的提升是better、faster、stronger，为此引入了passthrough操作保留细节信息，在一定程度上增强了小目标的检测能力。采用小卷积核替代7x7大卷积核，降低了计算量。同时改进的位置偏移的策略降低了检测目标框的难度。



# 参考文献

- [YOLO系列详解：YOLOv1、YOLOv2、YOLOv3、YOLOv4、YOLOv5、YOLOv6](https://blog.csdn.net/qq_40716944/article/details/114822515)
- [目标检测---IoU计算公式](https://blog.csdn.net/weixin_42206075/article/details/110471901)
- [YOLO家族进化史（v1-v7）](https://zhuanlan.zhihu.com/p/539932517)