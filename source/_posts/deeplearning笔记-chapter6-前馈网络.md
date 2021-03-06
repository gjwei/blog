---
title: deeplearning笔记-chapter6-前馈网络
date: 2018-05-12 10:21:01
tags: deeplearning笔记
---
# 第六章：深度前馈网络
也叫做多层感知器，是最典型的深度网络。

我们先从线性模型开始。

线性模型有着明显的缺陷，他的模型能力被限制在线性函数中，无法解释任何两个输入变量间的相互作用。
为了拓展线性模型表示非线性函数，可以讲输入x进行非线性的变化$\phi (x)$
<!-- more -->
如何选择映射函数呢？
1. 选择使用一种通用的映射函数，比如核函数。
2. 手动设计映射函数$\phi$
3. 深度学习的策略是学习一个映射函数$\phi$

这个让我突然想到了之前知乎上有人提问说，SVM和神经网络的关系。有人回答说，

SVM和神经网络都是为了解决感知器应用局限的问题，对于非线性的问题，SVM通过使用核函数等方法进行纠结，而神经网络采用的是多层感知器进行叠加解决。


## 实现XOR
我们使用两层的神经网络进行求解。可以看出，通过网络的stacking过程，可以大大拓展模型的数据拟合能力。
在更现实的应用中，学习的表示可以帮助模型泛化。

![](https://ws1.sinaimg.cn/large/9244e6f1gy1fr8dsbiuukj20bb0953yz.jpg)
在模型中使用ReLu作为非线性的激活函数
$$ h = ReLU(W^Tx + c) $$
可以致命整个网络是
$$ f(x;W,c,w,b)=w^T max(0, W^Tx + c) + b $$

可以得到一组解解：
![](https://ws1.sinaimg.cn/large/9244e6f1gy1fr8dwu8cxuj20ov08s0t7.jpg)

## 基于梯度的学习
神经网络的非线性导致了损失函数成为非凸的，意味着，模型的训练通常只能使得损失函数达到一个非常小的值，而不一定是全局最优解。

用于非凸损失函数的SGD没有收敛性的保证，对于参数的呃初始值十分敏感。


## 代价函数
多数情况下，模型定义了一个后验分布$p(y|x;\theta)$，我们可以通过使用训练数据和预测数据之间的交叉熵作为损失函数。即是*数据分布和模型之间的交叉熵**

对于深度学习的设计，反复出现的一个主题是**代价函数的梯度必须足够大和具有足够的预测性，为学习算法提供一个好的指引**。饱和的函数会破坏这一目的，因为他们的梯度变得非常小，更新的信息无法进行传递。

多数情况下，输出单元中会包含一一个指数函数，他的变量在取绝对值非常大的时候，梯度会出现饱和情况（梯度很小），而负对数似然中的对数函数就会消除输出单元中的指数效果，从而更好地传递梯度信息。

*均方误差和平均绝对误差在使用梯度下降的时候往往效果很差，一些饱和的输出单元在使用这些代价函数的时候会产生非常小的梯度。*

## 输出单元
### 用于高斯输出分布的线性单元
$$ y = Wh + b $$
线性输出层用来产生条件高斯分布的均值
$$ p(y|x) = N(y; \hat{y}, I) $$
最大似然概率此事等价于最小均方误差

### 用于伯努利分布的sigmoid单元
二值型变量y
sigmoid由于指数函数的存在，会出现梯度的饱和性，阻止基于梯度的学习做出好的改进。所以，要使用对数似然估计

### 用于多重伯努利分布的softmax单元
和sigmoid一样，softmax在极端负和极端正的情况下也会出现饱和。这个时候，基于梯度的损失函数也会饱和，学习将会停止。

softmax保证数值稳定的变体：
$$ softmax(z) =softmax(z - max(z)) $$

## 隐藏单元
如何选择合适的隐藏单元类型

整流线性单元是隐藏单元的默认选择。

多数情况下，隐藏单元都可以描述为接受一个输入向量x，计算仿射变换$z = W^tx + b$，然后使用一个逐元素的非线性函数g(z)。隐藏单元的差别仅仅在于激活函数g(z)的形式

### 整流线性单元及其拓展
ReLU:$g(z) = max(0, z)$



Leaky ReLU：

参数化ReLU

Maxout单元：
$$ g(z) = max z_j $$

maxout单元可以学习具有k个段的分段线性凸函数。
。。。

看到了很有意思的一句话：
LSTM通过求和在时间上传播信息。

## 架构设计
万能近似定理表明，无论我们试图学习什么样的函数，一个足够大的MLP一定可以表示这个函数。

单层的网络足以表示任意的函数，但是网络层可能大的不可实现，并且无法正确学习和泛化。使用更深的网络能够减少期望函数所需要的神经元个数已经具有更好的泛化性能。

更深的模型的泛化性能越好，但是也要考虑到数据量的问题。

## 反向传播

###  全连接MLP中的反向传播算法

#### 前向传播和损失函数的计算
![](https://ws1.sinaimg.cn/large/9244e6f1gy1fr8kyjjg3uj20jr0czjtc.jpg)

#### 反向传播过程
![](https://ws1.sinaimg.cn/large/9244e6f1gy1fr8l06qztlj20o50dmgo9.jpg)


## 历史小结
神经网络的核心思想是20世纪80年代提出的，到现在都没有发生重大改变。。近年来神经网络性能的改进可以归因于两个因素：
* 较大的数据集减少了统计泛化对神经网络的挑战的程度
* 更强大的计算机和更好的软件设施
* 少量的算法上的变化也显著的改善了神经网络性能

### 哪些算法的变化
1. 使用交叉熵族损失函数替代均方误差损失函数
2. 使用整流线性单元替代原有的sigmoid激活函数

### ReLU
从神经科学的角度去考虑
1. 对于某些输入，生物神经元是完全不活跃的
2. 对于某些输入，生物神经元的输出和输入成正比
3. 多数时间，生物神经元都是不活跃状态。（稀疏激活）