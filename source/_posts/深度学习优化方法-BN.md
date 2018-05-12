---
title: 深度学习优化方法-BN
date: 2018-05-03 21:53:40
tags:
- deeplearning
- NLP
---
# BN & LN 
批规范化的效果依赖于 minibatch 的大小，而且对于循环神经网络 RNN 无法下手. 本文将批规范化转换成层规范化——通过计算在一个训练样本上的一层上的神经元的求和输入的均值和方差.
<!-- more -->
有答案分析了一遍paper，那我来给个直观的理解，batch vs layer normalization。
batch是“竖”着来的，各个维度做归一化，所以与batch size有关系。
layer是“横”着来的，对一个样本，不同的神经元neuron间做归一化。



我的理解：
BN是在一个mini batch上进行归一化的，求解在所有batch上dim维度上的均值和方差，然后进行归一化
BN是要求每个batch输出的维度相同，但是RNN是处理变长的数据的，所以输出的维度是不同的。

LN是在一个样本上进行归一化，对样本的同一层的神经元输出进行归一化。所以，归一化的结果和batch没有关系