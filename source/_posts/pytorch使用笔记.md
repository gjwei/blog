---
title: pytorch使用笔记
date: 2018-05-30 21:50:52
tags: pytorch
---
## 优化器使用

### 为每个参数单独设置选项
<!-- more -->
```python
optim.SGD([{'params':  model.base.parameters()},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
```
这意味着 ``model.base`` 的参数将会使用 1e-2 的学习率,``model.classifier`` 的参数将会使用 1e-3 的学习率, 并且 0.9 的 momentum 将应用于所有参数.

```python
ignored_params = list(map(id, model.fc.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,
                     model.parameters())

optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': model.fc.parameters(), 'lr': 1e-2}
            ], lr=1e-3, momentum=0.9)
```

### 如何调整学习率
``class torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)``
通过 gamma 在每一个 epoch 里面的 step_size 设置每个参数组的初始学习率衰减变量. 当 last_epoch=-1, 设置初始 lr 为 lr.
```python
# Assume optimizer use 0.05 as lr
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90
。。。

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
for epoch in range(100):
    scheduler.step()
    train(...)
    validate(...)
```

``class torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)[source]``
和上个方法类似，知识LR的衰减是在到达milestones才开始的。gamma是衰减的系数

``class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)``
当一个指标已经停止提升时减少学习率.模型通常受益于通过一次2-10的学习停止因素减少学习率

这个调度程序读取一个指标质量 以及看到 ‘patience’ 的数量在一个 epoch 里面如果没有提升, 这时学习率已经减小.

```python
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> scheduler = ReduceLROnPlateau(optimizer, 'min')
>>> for epoch in range(10):
>>>     train(...)
>>>     val_loss = validate(...)
>>>     # Note that step should be called after validate()
>>>     scheduler.step(val_loss)
```