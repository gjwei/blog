---
title: paper-xgboost
date: 2018-05-14 14:49:23
tags: machine learning
---
# XGBoost: A Scalable Tree Boosting System
## 论文提出的贡献点：
1. 一个新的tree learning algorithm，可以有效处理sparse data
2. 理论上合理的加权分位数程序可以在近似树学习中处理实例权重。

贡献点：
1. 设计一个高可拓展性的end-to-end tree boosting系统
2. 理论上合理的加权分位数程序可以在近似树学习中处理实例权重。
3. 我们引入了一种新颖的并行树学习稀疏感知算法。
4. 我们针对树学习提出了一种有效的缓存感知块结构
5. 提出一个一个正则化的学习目标作为进一步的改进。

## Tree boosting算法

### 正则化的学习目标
![](https://ws1.sinaimg.cn/large/9244e6f1gy1frawcl51r6j20mt0cjtbs.jpg)

上图显示了一个Ensemble算法的过程，可以概括为一个加法模型。
![](https://ws1.sinaimg.cn/large/9244e6f1gy1frawmp9x8mj20lp055gm6.jpg)
和decision tree不同，每个回归树都会在叶子节点有一个连续的score值，我们只用$w_i$表示第i个节点的呃score值。

从而，就可以定义一个包含正则项的损失函数：
![](https://raw.githubusercontent.com/gjwei/images/master/20180514154137.png)
在这里$l$表示一个自定义的凸loss function。
加法的正则化项可以平滑最终学习到的权值，避免出现过拟合。

当去除了正则化项的时候，学习的目标就会变成传统的gradient tree boosting算法。

### Gradient Tree Boosting
