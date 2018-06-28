---
title: NCE_NS
date: 2018-06-26 15:27:56
tags: NLP, paper
---
# Notes on Noise Contrastive Estimation and Negative Sampling

## Abstract
在进行估计LM的参数估计的时候，因为需要评估整个字典的每个word的概率，所以计算起来非常困难。有两种相关的策略用来解决这个问题：Noise ccontrastive estimation 和 Negative Sampling
这篇论文指出了，尽管二者非常相似，但是NCE是渐进无偏的一半参数估计技术，而Negative sampling可以理解为一个二分类模型，他对于学习word representation非常有用，但是不能作为一个通用的方法。

## Introduction
对于LM需要估计下一个单词w。
![](https://raw.githubusercontent.com/gjwei/images/master/20180626153537.png)

和前面所叙述的一样，vocab太大，会导致计算概率非常困难。

NCE和NS方法都是将这种计算代价大的方法转化为二分类的代理问题，使用相同的参数，但是更容易进行计算。

## Noise contrastive estimation (NCE)
NCE将语言模型估计问题减少为估计概率二元分类器的参数的问题，该概率二元分类器使用相同的参数来区分样本与经验分布样本与噪声分布产生的样本

NCE方法两类数据产生过程如下：
1. 从真实数据分布p(c)中采样采样一个数据c，得到一个真实样本，标记为true

2. 从‘noise 分布 q(w)’选择k个noise samples ,标记为0.

对于给定正负样本的联合概率就可以表示为：

![](https://raw.githubusercontent.com/gjwei/images/master/20180626160323.png)

转转化为条件概率：
![](https://raw.githubusercontent.com/gjwei/images/master/20180626160417.png)
