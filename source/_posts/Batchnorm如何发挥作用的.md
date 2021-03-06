---
title: Batchnorm如何发挥作用的
date: 2018-07-16 17:45:55
tags: deep learning
---

# How Does Batch Normalization Help Optimization? (No, It Is Not About Internal Covariate Shift)


MIT的研究人员**从优化过程对应的空间平滑性**这一角度进行分析BN在深度学习中的作用

BatchNormalizatio是一种通过固定层输入的分布来改善神经网络训练的技术，通过引入一个附件的网络控制这些分布的均值和方差。

<!-- more -->

目前，对BatchNorm的成功以及其最初动机的最广泛接受的解释是，这种有效性源于在训练过程中控制每层输入数据分布的变化以减少所谓的“Internal Covariate Shift”。**那什么是Internal Covariate Shift呢，可以理解为在神经网络的训练过程中，由于参数改变，而引起层输入分布的变化。**研究人员们推测，这种持续的变化会对训练造成负面影响，而BatchNorm恰好减少了Internal Covariate Shift，从而弥补这种影响。

作者发现了BatchNorm对训练过程有着更根本的影响：**它能使优化问题的解空间更加平滑，而这种平滑性确保了梯度更具预测性和稳定性，因此可以使用更大范围的学习速率并获得更快的网络收敛**。

## BN究竟发挥了什么作用?

作者提出，BN是对训练过程的影响：通过对底层优化问题的再参数化，使得解空间更加平滑。损失函数的额平滑性得到改进，即损失函数能够以较小的速率进行变化，梯度的幅度也变小，然而效果更强。这种平滑性对训练算法的性能起到主要的影响。改进梯度的平滑性，可以在计算梯度方向上采用更大的步长。

它能使任何基于梯度的训练算法采取更大的步长之后，**防止损失函数的解空间突变，既不会掉入梯度消失的平坦区域，也不会掉入梯度爆炸的尖锐局部最小值**。这也就使得我们能够用更大的学习速率，并且通常会使得训练速度更快而对超参数的选择更不敏感。因此，是BatchNorm的平滑效果提高了训练性能。


