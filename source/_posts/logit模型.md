---
title: logit模型
date: 2018-07-11 17:08:37
tags: machine learning
---
## 解读logistics模型

### Odds
在统计学中，概率（Probability）和Odds都是用来描述某件事情发生的可能性。
Odds表示事情发生概率和不发生概率的比值。
<!-- more -->
![](https://raw.githubusercontent.com/gjwei/images/master/20180711171050.png)

我们知道，p的取值范围是[0, 1],所以，可以看出Odds的取值是0 -- 正无穷大

如果对Odds取自然对数，就可以将概率从[0, 1]映射到正负无穷大了
**Odds的对数称之为Logits**


### 和Logistics模型
对于分类任务，我们希望得到数据点属于某个类别的概率，如果使用线性模型做分类的时候，如何将线性模型方程和概率联系到一起呢？
就是通过Logits！！
通过令：
$$ logits = wx + b $$
从而，将线性方程和概率p映射到了一起。Logitics模型也叫做logits模型。