
# 1. R-CNN、Fast R-CNN、 Faster R-CNN区别

![[Pasted image 20230212180334.png]]

- R-CNN使用选择性搜索生成选框，它有一个缺点就是很慢，因为会在每个选框上应用cnn以提取特征，然后对提取的特征应用分类。
- Fast R-CNN在图像上应用一次卷积并在特征级别提取特征。其瓶颈在于区域特征提取。
- Faster R-CNN采用区域生成选框解决问题，它没有用选择性搜索，而用不同大小的锚框。

# 1. 结构

如图所示，Faster RCNN分为四个部分：

1. Conv layers。首先使用一组基础的conv+relu+pooling层提取image的feature maps。
2. Region Proposal Networks。区域选取网络。用于生成region proposals。通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。
3. Roi Pooling。收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。
4. Classification。利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。

![[Pasted image 20230212175145.png]]

图2展示了python版本中的VGG16模型中的faster_rcnn_test.pt的网络结构，可以清晰的看到该网络对于一副任意大小P×Q的图像：

1. 将P×Q缩放至M×N，送入网络；
2. Conv layers由13个conv层、13个relu层和4个pooling层组成，用于提取特征Map；
3. RPN网络首先经过3x3卷积，再分别生成positive anchors和对应bounding box regression偏移量，然后计算出proposals；
4. Roi Pooling层利用proposals从feature maps中提取proposal feature送入后续全连接和softmax网络作classification（即分类proposal到底是什么object）。

![[Pasted image 20230212175543.png]]

## 1.1 Conv layers

包含了conv，pooling，relu三种层。在Conv layers中：
1. 所有conv层都是：kernel_size=3，stride=1，pad=1
2. 所有的pooling层都是：kernel_size=2，stride=2，pad=0
一个MxN大小的矩阵经过Conv layers固定变为(M/16)x(N/16)。

## 1.2 Region Proposal Networks(RPN)

前面提到Faster RCNN使用RPN生成检测框，图上可以看出RPN网络分为两条线。
1. 上面一条通过softmax分类anchors获得positive和negative分类
2. 下面一条用于计算对于anchors的bounding box regression偏移量
Proposal层则负责综合positive anchors和对应bounding box regression偏移量获取proposals，同时剔除太小和超出边界的proposals。

### 1.2.1 anchors

anchors是一组由rpn/generate_anchors.py生成的矩形。`generate_anchors.py`中，相关代码会生成一个数组，每一个元素以数组的形式保存了矩形的左上和右下坐标。


![[Pasted image 20230212183435.png]]

anchors是为了保证多尺度，anchors size要根据检测图像设置的。原文中共有9个anchors，这是为了用作初始检测框。

anchors的产生过程：
1. 在feature map的每个中心点映射回原图，得到的是一个16\*16的区域。
2. 这16\*16的区域范围的中心点，以1：2，1：1，2：1的比例变换长宽比，再将区域的宽和高进行8，16，32三种倍数的放大。
3. 得到9个anchor。


![[Pasted image 20230212183838.png]]

1. Conv Layers中最后的conv5层num_output=256，对应生成256张特征图，所以相当于feature map每个点都是256-dimensions
2. conv5之后，做了rpn_conv/3x3卷积且num_output=256，相当于每个点又融合了周围3x3的空间信息
3. 假设在conv5 feature map中每个点上有k个anchor（默认k=9），而每个anhcor要分positive和negative，所以每个点由256d feature转化为cls=2•k scores；而每个anchor都有(x, y, w, h)对应4个偏移量，所以reg=4•k coordinates
4. 训练程序会在合适的anchors中**随机**选取128个postive anchors+128个negative anchors进行训练

这里有几个疑问：
1. 为什么是256维？
答：我们只需要一个3\*3\*256\*256这样的一个4维的卷积核，就可以将每一个3\*3的sliding window 卷积成一个256维的向量。
2. cls和coordinate？
答：
- cls（classify） layer 的作用是分类这块区域是前景还是后景，并分别打分，给出相应分数（也可以理解为相对概率）。
- reg(regression) layer 则是预测出这个box的x,y,w,h的位置（注意这里是相对于原图坐标的偏移）

>实际上，对于卷积层提取出来的feature map，在ZFNet中为13x13x256，作者使用了一个3x3的滑动窗口进行处理，其实就是一个3x3x256x256的卷积核（padding=1，stride=1），每个窗口的中心点，即图中的蓝点，会根据上面介绍过的Anchor生成机制，对应到原图上k=9个不同的矩形框。而得到的13x13x256的输出，就负责这13x13x9个矩形框的分类得分以及回归，具体分为两路：
>一路通过1x1x256x18的卷积核得到13x13x18的输出，也就是对于每个框预测其为前景和背景的得分；
>另一路通过1x1x256x36的卷积核得到13x13x36的输出，也就是对于每个框的大小和位置进行初步的回归。

>**其实RPN最终就是在原图尺度上，设置了密密麻麻的候选Anchor。然后用cnn去判断哪些Anchor是里面有目标的positive anchor，哪些是没目标的negative anchor。所以，仅仅是个二分类而已！**

### 1.2.2 softmax判定positive与negative

1×1卷积之后num_output=18，也就是输出图像为WxHx18大小。
也就是说feature maps每一个点都有9个anchors，同时每个anchors又有可能是positive和negative，所有这些信息都保存WxHx(9\*2)大小的矩阵。
后面接softmax分类获得positive anchors。

**为什么有两次reshape？**

其实是为了变换维度方便softmax分类，因为`blob=[batch_size, channel，height，width]`，而softmax分类需要二分类，所以reshape会将其变为[1,2,9×H,W]大小，之后再恢复原状。

### 1.2.3 bounding box regression

![[Pasted image 20230212192144.png]]

绿色框为飞机的Ground Truth(GT)，红色为提取的positive anchors，由于红色的框定位不准，这张图相当于没有正确的检测出飞机。所以我们希望采用一种方法对红色的框进行微调，使得positive anchors和GT更加接近。

对于窗口一般使用四维向量 (x,y,w,ℎ) 表示，分别表示窗口的中心点坐标和宽高。我们的目标是寻找一种关系，使得输入原始的anchor A经过映射得到一个跟真实窗口G更接近的回归窗口G'，即：

![[Pasted image 20230212192245.png]]

所以比较简单的思路是：

![[Pasted image 20230212192306.png]]

需要学习的是 $d_x(A) d_y(A) d_w(A) d_h(A)$这四个变换。当输入的anchor A与GT相差较小时，可以认为这种变换是一种线性变换， 那么就可以用线性回归来建模对窗口进行微调。

接下来的问题就是如何通过线性回归获得参数。线性回归就是给定输入的特征向量X, 学习一组参数W, 使得经过线性回归后的值跟真实值Y非常接近，即Y=WX。

(具体待补充)

### 1.2.4 对proposals进行bounding box regression

num_output=36，即经过该卷积输出图像为WxHx36，在caffe blob存储为[1, 4x9, H, W]，这里相当于feature maps每个点都有9个anchors，每个anchors又都有4个用于回归的变换量。

VGG输出50\*38\*512的特征，对应设置50\*38\*k个anchor，分别输出50\*38\*2k的positive/negative softmax分类特征矩阵和50\*38\*4k的regression坐标回归特征矩阵。

### 1.2.5 Proposal Layer

通过上面一系列的分类和回归操作以后，Anchor box变成了Proposal,就是RPN给出的建议框。但是其数量还是很庞大的，所以需要进行整理。
一般有以下几个步骤：
1. 生成anchors，利用变换量对所有的anchors做bbox regression回归（这里的anchors生成和训练时完全一致）
2. 按照输入的positive softmax scores由大到小排序anchors，提取前pre_nms_topN(e.g. 6000)个anchors，即提取修正位置后的positive anchors
3. 限定超出图像边界的positive anchors为图像边界，防止后续roi pooling时proposal超出图像边界
4. 剔除尺寸非常小的positive anchors
5. 对剩余的positive anchors进行NMS（nonmaximum suppression）非极大值抑制
6. Proposal Layer有3个输入：positive和negative anchors分类器结果rpn_cls_prob_reshape，对应的bbox reg的(e.g. 300)结果作为proposal输出

简而言之，就是通过映射回到原图input image，如果发现有proposal的边界超出原图范围，剔除严重超出的proposal；对剩下的边框softmax打分，再筛选出分数最高的proposal；再进行非极大值抑制，再排序，最终选择分数最高的proposal。

## 1.2 RoI pooling

负责收集proposal，并计算出proposal feature maps，送入后续网络。
Rol pooling层有2个输入：
1.  原始的feature maps
2.  RPN输出的proposal boxes（大小各不相同）
```python
```text
layer {
  name: "roi_pool5"
  type: "ROIPooling"
  bottom: "conv5_3"
  bottom: "rois"
  top: "pool5"
  roi_pooling_param {
    pooled_w: 7
    pooled_h: 7
    spatial_scale: 0.0625 # 1/16
  }
}
```

 - 由于proposal是对应MxN尺度的，所以首先使用spatial_scale参数将其映射回(M/16)x(N/16)大小的feature map尺度；
-  再将每个proposal对应的feature map区域水平分为 pooled_w×pooled_h 的网格；
-  对网格的每一份都进行max pooling处理。

![[Pasted image 20230212194322.png]]
这样就实现了固定长度的输出。

## 1.4 Classification

利用已经获得的proposal feature maps，通过full connect层与softmax计算每个proposal具体属于那个类别（如人，车，电视等），输出cls_prob概率向量；同时再次利用bounding box regression获得每个proposal的位置偏移量bbox_pred，用于回归更加精确的目标检测框。


![[Pasted image 20230212194407.png]]

# 2. 训练

Faster R-CNN的训练，是在已经训练好的model（如VGG_CNN_M_1024，VGG，ZF）的基础上继续进行训练。实际中训练过程分为6个步骤：

1.  在已经训练好的model上，训练RPN网络，对应stage1_rpn_train.pt
2.  利用步骤1中训练好的RPN网络，收集proposals，对应rpn_test.pt
3.  第一次训练Fast RCNN网络，对应stage1_fast_rcnn_train.pt
4.  第二训练RPN网络，对应stage2_rpn_train.pt
5.  再次利用步骤4中训练好的RPN网络，收集proposals，对应rpn_test.pt
6.  第二次训练Fast RCNN网络，对应stage2_fast_rcnn_train.pt

![[Pasted image 20230212194500.png]]

# 参考文献


- [一文读懂Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)
- [[DeepReader] Faster R CNN](https://www.youtube.com/watch?v=ixg4tlPL3X4)
- [faster rcnn中rpn的anchor，sliding windows，proposals？](https://www.zhihu.com/question/42205480#:~:text=anchor%20%E8%AE%A9%E7%BD%91%E7%BB%9C%E5%AD%A6%E4%B9%A0%E5%88%B0%E7%9A%84%E6%98%AF%E4%B8%80%E7%A7%8D%E6%8E%A8%E6%96%AD%E7%9A%84%E8%83%BD%E5%8A%9B%E3%80%82%20%E7%BD%91%E7%BB%9C%E4%B8%8D%E4%BC%9A%E8%AE%A4%E4%B8%BA%E5%AE%83%E6%8B%BF%E5%88%B0%E7%9A%84%E8%BF%99%E4%B8%80%E5%B0%8F%E5%9D%97%20feature%20map%20%E5%85%B7%E6%9C%89%E4%B8%83%E5%8D%81%E4%BA%8C%E5%8F%98%E7%9A%84%E8%83%BD%E5%8A%9B%EF%BC%8C%E8%83%BD%E5%90%8C%E6%97%B6%E4%BB%8E%209%20%E7%A7%8D%E4%B8%8D%E5%90%8C%E7%9A%84,%E6%8B%A5%E6%9C%89%20anchor%20%E7%9A%84%20rpn%20%E5%81%9A%E7%9A%84%E4%BA%8B%E6%83%85%E6%98%AF%E5%AE%83%E5%B7%B2%E7%9F%A5%E5%9B%BE%E5%83%8F%E4%B8%AD%E7%9A%84%E6%9F%90%E4%B8%80%E9%83%A8%E5%88%86%E7%9A%84%20feature%EF%BC%88%E4%B9%9F%E5%B0%B1%E6%98%AF%E6%BB%91%E5%8A%A8%E7%AA%97%E5%8F%A3%E7%9A%84%E8%BE%93%E5%85%A5%EF%BC%89%EF%BC%8C%E5%88%A4%E6%96%AD%20anchor%20%E6%98%AF%E7%89%A9%E4%BD%93%E7%9A%84%E6%A6%82%E7%8E%87%E3%80%82)
- [RPN（Region Proposal Network）和 Anchor 理解](https://blog.csdn.net/qq_35586657/article/details/97956189)
- [RPN的功能实现（流程与理解）](https://blog.csdn.net/w437684664/article/details/104238521)
- [捋一捋pytorch官方FasterRCNN代码](https://zhuanlan.zhihu.com/p/145842317)