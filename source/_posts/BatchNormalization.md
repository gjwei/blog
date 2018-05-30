---
title: BatchNormalization
date: 2018-05-30 13:38:04
tags: deeplearning
---
![](https://ws1.sinaimg.cn/large/9244e6f1gy1frtbf4y6ayj20ku0gddn6.jpg)
<!-- more -->
## BatchNormalization是什么？
BN的算法如上图所示，首先对某一层神经元的输出进行normalize，然后通过设置两个可以学习的参数进行scale and shift。
强制进行归一化会导致模型表达能力的降低，使用scale and shift是为了保证BN能够有能力恢复成原有的输入形式，保证了模型的表达能力。

对于全连接的网络，BN是对每一个神经元的输出进行批标准化的。而对于CNN这种共享权值的特殊结构，BN是采用空间 BatchNormalization的方法，做法是对每个kernel的所有输出进行批标准化处理。

## 为什么要有BN？
BN的提出是为了解决神经网络训练中会出现的internal covarite shift的问题。这个是指，在训练过程中，每一层神经元参数更新的假设都是基于其他层神经元参数没有变化的前提，然而事实上，这个假设是不成立的。
批标准化通过强制的重参数化输出，解决了这个问题。

### BN本质上是解决了梯度消失的问题
参考链接：https://www.zhihu.com/question/38102762
首先，解释下为什么会有梯度消失
![](https://raw.githubusercontent.com/gjwei/images/master/20180530152954.png)

BN的作用是对输出进行重参数化
$$ h_l = BN(w_lh_{l-1}) = BN(\alpha w_lh_{l-1}) $$
进行反向求导的时候可以得到：
![](https://raw.githubusercontent.com/gjwei/images/master/20180530153153.png)

在这里，我们可以将$BN(\alpha w_l h_{l-1})$简化成$\alpha w_l h_{l-1}$
这个时候，从l层传到k层的梯度信息就可以写成：
$$ \nabla_{h_k} l = \nabla_{h_l} l \prod_{i=k+1}^l{\alpha_i w_i} $$

所以，通过控制alpah的大小，就可以解决梯度消失的问题。


## BN的优势

1. 解决了internal covariate shift问题，可以使用更大的学习率
2. 减少了梯度消失的问题
3. 对初始化要求降低了。
![](https://raw.githubusercontent.com/gjwei/images/master/20180530152706.png)
参数提高了k倍，输出不变。


