
CV能解决图像分类、检测、分割的问题。

- 图像分类：输入图像，输出类别
- 目标检测：输入图像，输出多个类别的多个框
- 图像分割：语义分割和实例分割
	- 语义分割：对每一个像素分类，不管这个像素是属于哪个物体的，只管它是属于哪个类别的。（并不区分每个物体）
	- 实例分割：区分同一个类别的不同实例。

目标检测的发展主要分为两个分支：
两阶段是先从图像中提取若干候选框，再逐一对其分类及调整坐标，最后得出结果。
YOLO属于单阶段模型，不提去候选框，直接把全图feed到模型中，直接输出结果。

![[Pasted image 20230222130200.png]]

YOLO是当前目标检测性能最优的模型之一，所以非常重要。


# 1. mAP相关概念

## 1.1 正例与负例

现在假设我们的分类目标只有两类，计为正例（positive）和负例（negtive），然后我们就能得到如下的四种情况：

（1）True positives(TP):  被正确地划分为正例的个数，即实际为正例且被分类器划分为正例的实例数（样本数）；

（2）False positives(FP): 被错误地划分为正例的个数，即实际为负例但被分类器划分为正例的实例数；

（3）False negatives(FN):被错误地划分为负例的个数，即实际为正例但被分类器划分为负例的实例数；

（4）True negatives(TN): 被正确地划分为负例的个数，即实际为负例且被分类器划分为负例的实例数。

## 1.2 P（精确率）

P 代表 **precision**，即精确率，精确率表示**正确预测的正样本数占所有预测为正样本的数量的比值**，计算公式为：

**精确率 = 正确预测样本中实际正样本数 / 所有的正样本数**

即 **precision = TP/（TP+FP）**；

## 1.3 R（召回率）

R 代表 **recall** ，即召回率，召回率表示**正确预测的正样本数占真实正样本总数的比值**，计算公式为：  

**召回率 = 正确预测样本中实际正样本数 / 实际的样本数**

即 **Recall = TP/(TP+FN)** ；

## 1.4 ACC（准确率）

ACC 代表 **Accuracy**，即准确率，准确率表示**预测样本中预测正确数占所有样本数的比例**，计算公式为：

**准确率 = 预测样本中所有被正确分类的样本数 / 所有的样本数**

即 **ACC = （TP+TN）/（TP+FP+TN+FN）**；

IOU（Intersection over Union）：交并比，指的是ground truth bbox与predict bbox的交集面积占两者并集面积的一个比率，IoU值越大说明预测检测框的模型算法性能越好，通常在目标检测任务里将IoU>=0.7的区域设定为正例（目标），而将IoU<=0.3的区域设定为负例（背景），其余的会丢弃掉


AP（Average Percision）：AP为平均精度，指的是所有图片内的具体某一类的PR曲线下的面积，其计算方式有两种，第一种算法：首先设定一组recall阈值[0, 0.1, 0.2, …, 1]，然后对每个recall阈值从小到大取值，同时计算当取大于该recall阈值时top-n所对应的最大precision。这样，我们就计算出了11个precision，AP即为这11个precision的平均值，这种方法英文叫做11-point interpolated average precision；第二种算法：该方法类似，新的计算方法假设这N个样本中有M个正例，那么我们会得到M个recall值（1/M, 2/M, …, M/M）,对于每个recall值r，该recall阈值时top-n所对应的最大precision，然后对这M个precision值取平均即得到最后的AP值。



# 2. YOLOv1

## 2.1 论文思想

### 2.1.1 图像分为网格，如果一个object中心落在某网格中就用该网格预测该object



### 2.1.2 每个网格预测C个类别分数和B个bounding box，每个bounding box 除了要预测位置外还要附带预测一个confidence值。

- Bounding box预测五个值，四个是位置参数（x, y, w, h）。
- x，y是当前格子预测得到的物体的bounding box的中心位置的坐标，它被限制在该格子中间，因此x，y数值应该在[0, 1]。
- w，h是相对图像而言的，因此也在[0, 1]。
- confidence是预测目标与真实目标的交并比：$Pr(Object) * IOU^{truth}_{pred}$，Pr(Object)可以简单理解为0和1，当网格存在目标则为1，IOU是预测boundingbox与真实bounding box的交并比。
- 测试时 $Pr(Class_i|Object) * Pr(Object) * IOU^{truth}_{pred} = Pr(Class_i)* IOU^{truth}_{pred}$，  $Pr(Class_i|Object)$就是预测的c个类别分数中的第i个。
- 给出的概率分数既包含它为某个目标的概率，又包含预测的边界框与某个真实的边界框的重合程度。


![[Pasted image 20230220214049.png]]

## 2.2 网络结构


![[Pasted image 20230220220031.png]]

YOLO检测网络包括24个卷积层和2个全连接层，卷积层用来提取图像特征，全连接层用来预测图像位置和类别概率值。1\*1的卷积主要是为了降低特征空间，也为了跨通道的信息整合。
- transpose存不存在得看tensor，展平处理是为了和全连接层连接。
- 然后reshape成7×7深度为30的特征矩阵，可以理解为每一个cell对应的特征信息。

## 2.3 损失函数

 ![[Pasted image 20230220220741.png]]

- 为什么宽高使用开平方？对于大目标和小目标，如果偏移相同大小，对于小目标而言IOU更小，更无法接受。如果使用真实的宽高的话，比如y=x，对于大目标和小目标而言，如果x的差值相同，误差的影响也相同，显然是不合理的。
- obj-正样本损失，noobj-负样本损失（第i个grid cell中没有目标）。对于正样本，真实值=1；负样本，真实值=0。

## 2.4 问题

1. **对群体性的小目标检测效果**较差，因为每个grid cell只预测**两个同类别**的bounding box，对每个grid cell 只预测一组classes参数，所以每个grid cell预测同一个类别的目标，当一些小目标聚集在一起时检测效果会比较差。也就是如果同一个格子包含多个物体的时候，只会预测一个IOU最高的bounding box。 
2. 当目标出现新尺寸或目标比例时效果较差 
3. 因为采取直接预测目标坐标信息，所以主要的错误的原因都来自于**定位不准确**

# 3. YOLOv2

主要贡献：缩小定位误差，增加recall

## 3.1 批处理标准化

<font color='red'><b>目的：</b></font> CNN在训练过程中网络每层输入的分布一直改变，会使训练过程难度加大。对网络的每一层的输入(每个卷积层后)都做了归一化，**这样网络就不需要每层都去学数据的分布，收敛会更快**。

<font color='red'><b>方法：</b></font> 在所有卷积层后+BN层。

![[Pasted image 20230221144114.png]]

<font color='red'><b>效果：</b></font>

- 卷积层后加的BN层，有助于训练收敛，减少正则化，帮助正则化模型。
- 可以移除dropout层（防止过拟合）。
- mAP 提升2%。

## 3.2 高分辨率分类器

<font color='red'><b>目的：</b></font> 减少性能损失。

>v1中预训练使用的是分类数据集，大小是224×224 ，微调时使用YOLO模型做目标检测的时候才将输入变成448 × 448。这样改变尺寸，网络就要多重新学习一部分，会带来性能损失。

<font color='red'><b>方法：</b></font>v2直接在预训练中输入的就是**448×448**的尺寸，微调的时候也是**448 × 448**。

<font color='red'><b>效果：</b></font>mAP 提升4%。

## 3.3 Convolutional With Anchor Boxes—带有Anchor Boxes的卷积

<font color='red'><b>目的：</b></font> 简化目标边界框预测的问题。

> Anchor（先验框） 就是一组预设的边框。先大致在可能的位置“框”出来目标，然后再在这些预设边框的基础上进行调整。

<font color='red'><b>方法：</b></font>使用基于Anchor的目标边界框的预测，使用Anchor Box的Scale来表示目标的大小，使用Anchor Box的Aspect Ratio来表示目标的形状；删除全连接层和最后的pooling层以获得更高分辨率；缩小网络输入图像为416×416.

> 416×416在下采样之后得到特征图大小为13×13（stride = 32，416/32=13）。

<font color='red'><b>效果：</b></font>mAP下降0.3%，但是增加了7%的recall。

![[Pasted image 20230221145201.png]]


## 3.4 维度聚类

>使用Anchor的问题1：Anchor Boxes的尺寸是手工指定了长宽比和尺寸，相当于一个超参数，这违背了YOLO对于目标检测模型的初衷，**因为如果指定了Anchor的大小就没办法适应各种各样的物体了**。

<font color='red'><b>目的：</b></font>自动找到好的先验参数。

<font color='red'><b>方法：</b></font>基于训练集中所有目标的边界框用k-means聚类的方法获得相应的priors。

<font color='red'><b>效果：</b></font>使网络更容易学习预测好的检测结果。



## 3.5 Direct location prediction—直接的位置预测


> 使用Anchor的问题2：模型不稳定

<font color='red'><b>目的：</b></font>自动找到好的先验参数。

如果直接使用基于Anchor的预测，那么训练模型的时候会出现模型不稳定，大部分都来源于预测目标边界框中心坐标（x, y）导致。
$$x =(t_x * w_a)+x_a$$
$$y =(t_y * h_a)+y_a$$
这里$(t_x, t_y)$表示需要根据预测的坐标偏移值，$(w_a, h_a)$表示先验框anchor的宽高，$(x_a, y_a)$表示先验框的中心坐标。

![[Pasted image 20230221100354.png]]

就比如这个紫色框中的物体由左上角的anchor去预测，但是加上偏量之后可能跑到右下角（橙色框）去了，显然是不合理的。因为就算那边有物体也是由左上角的框（红色小框）进行预测。

<font color='red'><b>方法：</b></font> 预测边界框中心点相对于对应cell左上角位置的相对偏移值。将网格归一化为1×1，坐标控制在每个网格内，同时配合sigmod函数将预测值转换到0~1之间的办法，做到每一个Anchor只负责检测周围正负一个单位以内的目标box。

$(c_x, c_y)$是当前cell左上角到图像左上角的距离，bounding box prior宽高是$p_w, p_h$，那么有公式为：
$$b_x = \sigma(t_x) + c_x $$
$$b_y = \sigma(t_y) + c_y $$
$$b_w = p_we^{t_w} $$
$$b_h = p_he^{t_h} $$
$$Pr(object) * IOU(b, object) = \sigma(t_o)$$

$\sigma$就是sigmoid函数，预测中心点就会在该grid cell中间。
$Pr(object) * IOU(b, object)$表示预测边框的置信度，相比于yolov1直接预测置信度的值，这里对$t_o进行\sigma$ 变换之后获得置信度。

<font color='red'><b>效果：</b></font>mAP提升了约5%。

![[Pasted image 20230221111638.png]]

## 3.6 Fine-Grained Features—细粒度的特征

<font color='red'><b>目的：</b></font>追求对小对象的检测效果。

<font color='red'><b>方法：</b></font>加了一个passthrough layer，将相邻的特征叠加到不同的通道。将前层26×26×512的特征图转换为13×13×2048的特征图，并与原最后层特征图进行拼接。

比如 4 × 4特征图抽取每个 2 × 2局部区域组成新channel，也就是宽高/2，channel×4。

![[Pasted image 20230221152459.png]]

![[Pasted image 20230221152648.png]]

这边的步骤是：
1. 提取Darknet-19最后一个maxpooling层的输入，得到26×26×512的特征图。
2. 经过1×1×64的卷积以降低特征图的维度，得到26×26×64的特征图。
3. 经过pass through层的处理变成13x13x256的特征图。
4. 再与13×13×1024大小的特征图连接，变成13×13×1280的特征图。

<font color='red'><b>效果：</b></font>mAP提升了1%。

## 3.7 Multi-Scale Training—多尺度的训练

<font color='red'><b>目的：</b></font>改进yolov1448×448的固定分辨率输入，使得网络使用不同大小的输入图像。

<font color='red'><b>方法：</b></font> 每10batches训练后随机选择一个新的图像尺寸大小。由于模型下采样了32倍，从以下32的倍数{320,352，…，608}作为图像维度的选择。

<font color='red'><b>效果：</b></font>同一个网络可以进行不同分辨率的检测任务。小尺寸图片检测中成绩较好。

## 3.8 Backbone：Darknet-19

分类网络：19个卷积层+5个maxpooling层。

![[Pasted image 20230221153338.png]]

## 3.9 Training for detection —— 检测的训练

![[Pasted image 20230221153459.png]]

<font color='red'><b>方法：</b></font> 
- 移除最后一个卷积层、avgpooling层和softmax
- 增加3个3x3x1024的卷积层
- 增加1个1x1的卷积层。输出的channel数为num_ anchors×(5+num_ calsses)。
> 对于VOC，预测5个边界框，每个边界框有5个参数（4个坐标1个confidence），每个边界框有20个类，所以最后一个1×1卷积层有125个卷积核。
> (5 + 20) x 5。这里的类别是属于box的，和yolov1不一样。

- 倒数第1、2个卷积层之间加passthrough层，可以利用细粒度特征。

## 3.10 网络训练细节

思考：如何匹配正负样本？如何计算误差？

- 使用数据增强处理，随机裁剪、颜色增强等。

# 4. YOLOv3

- Darknet-53

因为有53个卷积层！

![[Pasted image 20230221160623.png]]

每个方框对应的是残差结构，主分支上是1x1卷积层和3x3卷积层，再将捷径分支上的输出从输入引过来与主分支上的输出相加。

## 4.1 目标边界框的预测

三个特征图中分别通过( 4 + 1 + c ) × k个大小为1 × 1 的卷积核进行预测，k为预设边界框（bounding box prior）的个数（在每个预测特征层中k默认取3），c为预测目标的类别数，

>N x N x \[ 3 * (4 + 1 + 80)] 

## 4.2 模型结构

![[Pasted image 20230221163204.png]]

Predict one
- Convolutional Set 是图右边的五个卷积层堆叠在一起
- 最后使用1 x 1大小的预测器进行预测，所以它的输出就是 13 x 13 x \[ 3 * (4 + 1 + 80)] 

往下
- 存在一个上采样层，宽高扩大为原来的两倍，变成26 x 26
- 和倒数第二个残差结构的输出拼接，是在深度上拼接

Predict two and three同理.

由于特征图的输出维度为N x N x \[ 3 * (4 + 1 + 80)] ，所以特征图1（13 x 13）用来预测大尺寸物体，图2（26 x 26）用来预测中尺寸物体，图3（52 x 52）用来预测小尺寸物体


## 4.3 正负样本匹配


>对每个GT（ground truth）分配一个bounding box prior.

也就是说对每个gt都分配一个正样本。图片中有几个gt目标就有几个正样本。

> 分配的原则就是将与gt重合度最高的bounding box prior作为正样本。如果一个bounding box prior不是最好但是超过了某个阈值，那就直接丢弃这个结果。

剩下的样本都为负样本。

> 如果一个bounding box prior没有分配给ground truth的话，既没有定位损失也没有类别损失，只有confidence score。

![[Pasted image 20230221165626.png]]

按照这个思想，会计算GT与三个Anchor模板的IOU值，然后设定一个阈值，大于这个值的AT就为正样本。

> 如果都大于，是不是把GT分配给三个AT？是的！可以扩充正样本数量。

## 4.4 损失计算

![[Pasted image 20230221165849.png]]

### 4.4.1 置信度损失

使用的二值交叉熵损失。

![[Pasted image 20230221165929.png]]

如果蓝色代表Anchor，绿色-gtbox，通过预测的偏移量运用到anchor上就得到黄色框。

### 4.4.2 类别损失

使用的二值交叉熵损失。

![[Pasted image 20230221170150.png]]

### 4.4.3 定位损失

使用了sum of squared error loss。tx网络预测中心点的回归参数

![[Pasted image 20230221170613.png]] 

# 5. yolov3-spp版

四个方面的尝试：Mosaic图像增强、SPP模块、CIOU Loss、Focal loss

## 5.1 Mosaic图像增强

就是使用多张图片拼接进行增强。

优点：
- 增加数据的多样性
- 增加目标个数
- BN能一次性统计多张图片的参数

> Batch size 尽量设置大一点，因为BN层主要求均值和方差，如果size越大，求得的均值和方差就越接近整个数据集的均值和方差。

将四张图片拼起来相当于batch size = 4

## 5.2 SPP模块

和YOLOv3的差别仅仅在于Darknet53输出和预测特征层1之间加上了一个SPP结构。

![[Pasted image 20230221204657.png]]

![[Pasted image 20230221204828.png]]

concatenate的是四个输出，一个是直接从输入接到输出，另外三个是经过最大池化下采样得到的。因此深度 x 4，变为16 x 16 x 2048。

问题：为什么只在第一个预测特征层之前接入SPP结构呢？

因为只添加一个就够了，加很多会提高推理速度。

![[Pasted image 20230221205158.png]]

## 5.3 CIoU Loss

v3里是插值平方，也就是L2损失。

CIoU Loss的发展：
IoU Loss -> GIoU Loss -> DIoU Loss -> CIoU Loss

### 5.3.1 IoU Loss

![[Pasted image 20230221205426.png]]

引入IoU Loss的原因是l2损失不能很好地反映重合程度。

### 5.3.2 GIoU Loss

![[Pasted image 20230221205620.png]]
绿色-真实目标边界框 红色-预测目标边界框 蓝色-$A^c$ u-绿红并集

引入GIoU可以解决不相交时loss为0的问题。

缺点：宽高一样且水平或垂直相交，GIoU退化成IoU。

![[Pasted image 20230221210031.png]]

### 5.3.3 DIoU Loss

![[Pasted image 20230221210127.png]]
左图上图：黑色-anchor 绿色-default box/真实目标边界框 蓝色-预测目标边界框
右图：IoU和GIoU都有缺陷。

前面两个的缺点：收敛慢、回归不精确

![[Pasted image 20230221210431.png]]
黑框-预测目标边界框 绿色-真实目标边界框的中心坐标
$\rho - b 和 b^{gt}$ 之间的欧式距离 ，b-预测目标边界框的中心坐标，$b^{gt}$-真实目标边界框的中心坐标
d-两个中心的距离，c-最小外接矩形的对角线长度

### 5.3.4 CIoU Loss

![[Pasted image 20230221210921.png]]


## 5.4 Focal Loss

![[Pasted image 20230221211206.png]]

问题：为什么two-stage网络没有类别不平衡的问题？

第一阶段肯定还会存在这个问题，但是可以通过第二阶段去确定目标的最终坐标。在Faster R-CNN提供给第二阶段的样本数就2k多个，比这上万个上十万个的好得多。

比如在图片中匹配到50个正样本，每个贡献的损失=3，对于每个负样本而言，可能损失很小，但因为数量很大所以会提供很大的损失。这就是**degenerate models**。

之前存在**hard negative mining** 选取损失比较大的负样本训练我们的网络，效果也比较好。

问题：为什么FL效果好？

![[Pasted image 20230221211824.png]]

### 5.4.1 平衡交叉熵

引入$\alpha$平衡样本比例。

![[Pasted image 20230221211945.png]]

### 5.4.2 Focal Loss Definition

![[Pasted image 20230221212055.png]]

损失函数，用于降低简单样本的权重。

### 5.4.3 最终形式

![[Pasted image 20230221212255.png]]

缺点：易受噪音干扰

# 6. YOLOv4

网络结构：
- Backbone: CSPDarknet53
- Neck: SPP, PAN
- Head: YOLOv3

优化策略：
- Eliminate grid sensitivity
- Mosaic data augmentation
- IoU threshold(match posotive samples)
- Optimizered Anchors
- CIOU

## 6.1 网络结构

- 增强CNN学习能力
- 移除计算瓶颈
- 降低显存使用

### 6.1.1 CSPDenseNet

![[Pasted image 20230221213037.png]]

![[Pasted image 20230221213104.png]]
![[Pasted image 20230221213236.png|275]]
DownSample345都一样，区别在于ResBlock通道不一样。

### 6.1.2 SPP

![[Pasted image 20230221213415.png]]

### 6.1.3 PAN

FPN 高层的语义信息往低层融合，又增加了低层向高层融合的部分。

![[Pasted image 20230221213638.png]]

PAN中的激活函数都是Leaky。

![[Pasted image 20230221213946.png]]

## 6.2 优化策略

### 6.2.1 Eliminate grid sensitivity

[[#4.4.3 定位损失]]

回顾一下Bounding boxes with dimension priors and location prediction，这里会有一个问题，就是如果中心点落在当前gril cell的边界，我们就会希望tx和ty都等于0。但是对于sigmoid函数，只有x趋近负无穷时y才趋近于0。

![[Pasted image 20230221214510.png]]

在相同的x范围内，缩放之后y对x更敏感。

### 6.2.2 Mosaic data augmentation

和前文一样

### 6.2.3 IoU threshold(match posotive samples)

[[#4.3 正负样本匹配]]

训练的时候正样本个数很少。

YOLOv4通过降低grid敏感度的方式，将相对gril左上角的偏移量从[0.5, 1]缩放到[-0.5, 1.5]之间。

达到了扩充正样本的目的。

![[Pasted image 20230221214842.png]]

### 6.2.4 Optimizered Anchors

![[Pasted image 20230221215150.png]]


### 6.2.5 CIOU

# 7. YOLOv5

## 7.1 网络结构

- Backbone: New CSP-Darknet53
- Neck: SPPF, New CSP-PAN
- Head: YOI Ov3 Head

Focus模块替换成6 x 6普通卷积层。

![[Pasted image 20230223125005.png]]

SPPF更快！

## 7.2 数据增强

Mosaic、Copy paste、Random affine、MixUp、Albumentations（滤波、直方图均衡化以及改变图片质量等等）

Augment HSV 调整色度 饱和度 明度

Random horizontal flip

## 7.3 训练策略

多尺度训练（0.5~1.5x）
AutoAnchor（For training custon data）自动根据数据集中的目标聚类生成新的anchor
Warmup and Cosine LR scheduler
EMA给学习变量增加动量
Mixed precision
Evlove hyper-parameters

## 7.4 损失计算

![[Pasted image 20230223125617.png]]

平衡不同尺度损失

![[Pasted image 20230223125702.png]]

消除Grid敏感度

![[Pasted image 20230223125913.png]]

![[Pasted image 20230223125931.png]]
如果采用之前的计算公式，e^x不受限。

匹配正样本

选择宽度高度方向差异最大的样本，如果rmax < anchor_t 那么就是匹配成功了。

![[Pasted image 20230223130141.png]]

然后就跟v4一样了。

# 8. YOLOX

anchor-free

decoupled detection head

advanced label assigning strategy(SimOTA)

## 8.1 网络结构

YOLOx由input、main network、neck和prediction四个部分组成。在输入端使用了Mosaic和MixUp作为数据增强策略，得到多个大小合适的图像。这是两种强大的增强策略，已经在多个研究中得以验证。主干网络使用了CSPDarknet53的网络结构，包含Focus结构、Center and Scale Prediction (CSP) network 和SPP，用于提取图像上的特征。Focus结构会对图片进行切片，将空间信息传输到输入图像上的通道维度，进行更快的推断。CSPNet通过将梯度的变化集成到特征图中，在减少了计算量的同时可以保证准确率。SPP则用于增大感受野。YOLOX还引入残差网络residual，形成更大的残差结构块，有助于解决梯度消失问题。另外，YOLOX采用SiLU激活函数，它的效果要好于ReLU。 Neck部分使用了FPN+PAN，用于接收CSPDarknet53的输出，进行不同尺度的特征融合并增强特征提取。Neck部分输出三个不同尺度的特征图到预测头。在预测头中，使用解耦头对分类和回归任务分别进行卷积，然后整合在一起预测。

## 8.2 Anchor-free

区别是wh不用 x anchor宽高

![[Pasted image 20230223131428.png]]

## 8.3 损失函数

![[Pasted image 20230223131541.png]]

正负样本匹配SimOTA

![[Pasted image 20230223131646.png]]



# 参考文献

- [深刻理解目标检测中的mAP评估指标](https://zhuanlan.zhihu.com/p/254973280)
- [3.1 YOLO系列理论合集(YOLOv1~v3)](https://www.bilibili.com/video/BV1yi4y1g7ro/?spm_id_from=333.337.search-card.all.click&vd_source=1820d6ae5f24e03d1db236795220170a)
- [【YOLO系列】YOLOv2论文超详细解读（翻译 ＋学习笔记）](https://blog.csdn.net/weixin_43334693/article/details/129087464)
- [史上最通俗易懂的YOLOv2讲解](https://blog.csdn.net/shanlepu6038/article/details/84778770)
- [YOLOv4网络详解](https://www.bilibili.com/video/BV1NF41147So/?spm_id_from=333.999.0.0&vd_source=9ea40c1ac510f1e604555a3c8278ff94)
- [YOLOv5网络详解](https://www.bilibili.com/video/BV1T3411p7zR/?spm_id_from=333.999.0.0&vd_source=9ea40c1ac510f1e604555a3c8278ff94)
- [YOLOX网络详解](https://www.bilibili.com/video/BV1JW4y1k76c/?spm_id_from=333.337.search-card.all.click&vd_source=9ea40c1ac510f1e604555a3c8278ff94)