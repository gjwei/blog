---
title: focal loss
date: 2018-05-24 16:38:59
tags: deeplearning
---
# Focal loss

Focal loss的提出主要是为了解决在物品检测中，前景物品和背景类别的极度不平衡问题。

![](https://ws1.sinaimg.cn/large/9244e6f1gy1frnokl3iv7j20ls0k7wi7.jpg)
<!-- more -->
首先，介绍下交叉熵：

![](https://ws1.sinaimg.cn/large/9244e6f1gy1frnljzsp7nj20gg02zwel.jpg)

我们定义$p_t$为：
![](https://ws1.sinaimg.cn/large/9244e6f1gy1frnlkv1mm0j20ez03a749.jpg)
这样，交叉熵可以写成：
$$ CE(p, y) = CE(p_t) = -log(p_t) $$

从上图汇总可以看出，对于容易分类的情况（p >> 0.5)，也会产生一个不小的loss，如果大量累加这种easy example，这些small loss将会碾压少见的类。
```python
class CrossEntropy(nn.Module):
    def __init__(self, eps=1e-6):
        self.eps = eps

    def forward(self, preds, targets):
        # preds: b x num_classes
        # labels: b x 1
        y = one_hot(targets, preds.size(-1))
        preds = preds.clamp(self.eps, 1.0 - self.eps)
        loss = -1 * targets * torch.log(preds)
        return loss
```

### Balance CrossEntroy

![](https://ws1.sinaimg.cn/large/9244e6f1gy1frnoz7w84sj208a0290sl.jpg)
对于类别不均衡的情况，可以引入一个权值因子$\alpha$，用来调节学习的情况。
可以对类别少的设置大的错误惩罚权值，这样系统更不容易将类别少的对象识别错。

```python
class BalancedCrossEntropy(nn.Module):
    def __init__(self, alphas, eps=1e-6):
        # alphas: num_classes,
        super(BalancedCrossEntropy, self).__init__()
        self.eps = eps
        self.alpahs = torch.LongTensor(alphas)

    def forward(self, preds, targets):
        y = one_hot(targets):
        weights = self.alpahs[targets]
        preds = preds.clamp(self.eps, 1.0 - self.eps)
        loss  = -1 * targets * weights * torch.log(preds)
        return loss.sum()
```
### Focal Loss Deﬁnition
虽然使用$\alpha$平衡了postive/negative类别不均衡的问题，但是并没有区分出easy/hard的问题，我们通过重塑损失函数以减少简单例子的loss比例，从而让学习更集中在hard的负面例子上。

![](https://ws1.sinaimg.cn/large/9244e6f1gy1frnpn8zo7uj20ax021t8k.jpg)

在实际中，我们使用alpha-Balance的FL变种
![](https://ws1.sinaimg.cn/large/9244e6f1gy1frnpoa1j08j20b60233ye.jpg)

代码如下：
```python
class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum()
```