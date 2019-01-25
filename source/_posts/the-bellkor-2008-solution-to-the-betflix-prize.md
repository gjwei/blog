---
title: The BellKor 2008 Solution to the Netflix Prize
date: 2019-01-22 13:56:49
tags: Recommender System
---
# 介绍
主要介绍论文结构
# 新方法
## 因子模型
### SVD
![](https://raw.githubusercontent.com/gjwei/images/master/20190122140015.png)

### Asymmetric-SVD
![](https://raw.githubusercontent.com/gjwei/images/master/20190122140641.png)
### SVD++
![](https://raw.githubusercontent.com/gjwei/images/master/20190122140713.png)

这些模型的学习方法是使用SGD，参数$\mu$是指训练集中的平均打分值。

## 时间效应
发现了三种强时间效应
1. 电影偏差：电影会随着时间发生流行度变化
2. 用户偏差：用户会随着时间改变他们的打分基线
3. 用户偏好：用户的偏好会随着时间发生变化

之后，开始介绍下如何在模型中使用这些时间效应
以SVD++为例：
![](https://raw.githubusercontent.com/gjwei/images/master/20190122141612.png)
Here，我们预测随着时间t的打分值r。请注意，相关的参数现在构造为和时间相关的函数。这些偏差是同时学习的。
时间偏差效应更容易捕捉到，因为我们不需要最精细的细粒度。
对于item的时间效应，比较容易捕获到：
作者将训练集时间切分了30份，每一份作为一个bin。同一个电影在不同bin具有不同的结果。
![](https://raw.githubusercontent.com/gjwei/images/master/20190122150408.png)
对于用户层面的时间效应：
![](https://raw.githubusercontent.com/gjwei/images/master/20190122150500.png)
![](https://raw.githubusercontent.com/gjwei/images/master/20190122150511.png)
![](https://raw.githubusercontent.com/gjwei/images/master/20190122150558.png)

