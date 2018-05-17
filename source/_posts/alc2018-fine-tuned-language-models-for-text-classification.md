---
title: alc2018-fine-tuned-language-models-for-text-classification
date: 2018-05-17 10:57:08
tags: NLP
---
# Fine-tuned Language Models for Text Classiﬁcation
## 摘要
迁移学习已经对CV领域革新了，但是现有的NLP方法仍然需要特定的任务修改和从头开始训练。
论文提出一种Fine-tuned Language Models（FitLaM）,能够有效的对NLP任何任务进行迁移学习，并且介绍了使用微调技术得到的state-of-the-art的结果
<!-- more -->
## 介绍
主要是介绍下CV领域的迁移学习方法
一个成功的NLP迁移学习应该和CV一样有其对应的标准类似
1. 该方法可以利用大量的可用数据
2. 利用进一步的优化任务，导致进一步提高下游的任务
3. 可以依赖一个可以用作多数NLP任务的模型
4. 在实践中很容易使用


FitLaM依赖于一个简单的RNN网络，无需任何的修改，我们只用一个或者多个特定的任务的线性层增强模型，对现有的方法来说，只涉及s少数的参数更新。

同时，文章还提出一种新的微调技术，可以进行有差别的调整参数，低层次的层调节的幅度要小于高层次的层，已保留语言模型获得的知识。

### 论文贡献点
1. 提出一中NLP的有效迁移学习方法
2. 提出FitLaM，这是一种可以实现任意NLP任务的类CV迁移学习方法
3. 提出Discriminative Fine-Tuing方法，保留以前的知识，防止在微调期间发生灾难性遗忘
4. 引入了文本分类中的BPTT，一种通过线性层将分类器损失反向传播到任何序列大小的RNN的输出
5. 对微调训练LM介绍了一些关键的技术
6. 模型超越了五个公开数据集，多数数据集误差减少了18%-24%
7. 提供预训练模型和代码

## 相关工作
### CV领域的迁移学习

### Hypercolumns
在NLP中，最近有人提出一种没有使用word embedding进行迁移学习的方法。方法是将pretrain embeedding通过其他的任务捕获到上下文的context信息，然后将这种’embedding‘和word embedding以及中间层的输入连接到一起。

### Multi-task learning
MTL是将多个任务联合进行学习。

### Fine-Tuning
对预训练的模型进行微调

## Fine-tuned Language Models (FitLaM)
这个模型在一个大的通用领域语料上预训练高度优化的语言模型（LM),并将其调整到目标任务上。

LM试图通过前面的单词来预测下一个单词的概率。模型依赖于使用不同的方法（从n-gram到RNN）的语料库联合概率的自回归因式分解，这些方法在基准测试中都达到了state-of-the-art的结果。在实验中，使用最先进的语言模型**（AWD-LSTM）**，这是一个具有很强正则化策略的正则LSTM。类似于CV，我们将会使用更高性能的语言模型改善下游任务。

LitLaM包含以下步骤：
1. 通用领域的LM预训练
2. 对目标任务LM微调
3. 目标任务分类器微调

### 通用领域的LM预训练
在Wikitext-103，共有28595篇预处理好的Wikipedia文章。
通过这些数据集预训练LM模型，然后将模型的参数保存下来，作为下游任务的预训练模型。

### Target task LM fine-Tuning
使用之前的单词，预测下一个单词。

目标任务的数据可能来自于不同的分布，所以，需要对目标任务的LM进行优化。在给定的通用LM模型上进行训练，这个阶段的收敛速度要快很多，因为它只需要适应目标数据的优化，并且允许使用小的数据集也能够训练一个强大的LM。

#### Gradual unfreezing
逐渐解冻方法，实验表明从最后一层开始逐渐解冻的模型最有用，这是因为这层包含最少的一般知识。
首先，解冻最后一层，然后微调所有的未解冻层。
然后，解冻下一层，重复这个过程。知道所有的层都进行了精细的调整。

注意，我们进行训练的时候，是每次都会解冻一层，然后训练所有解冻的层，而不是一次只训练一层。

#### Fine-tuning with cosine annealing 使用余弦退火的方法进行微调
这种方法经过试验效果最好。
我们只训练一个epoch，并且按照下面的schedule在每个batch进行降低学习率的

![](https://raw.githubusercontent.com/gjwei/images/master/20180517152338.png)

在这里$n_t$表示step t对应batch的学习率。

#### Warm-up reverse annealing
实验发现，在训练所有层的时候增加学习率很有用。

#### Target task classiﬁer ﬁne-tuning
对于分类任务进行fine tuning
对于分类任务，我们采用对LM增加一层或者更多的线性block。

每个block都是使用BN，dropout，relu以及最后输出到一个softmax函数中。注意这些任务中，只有一层的参数需要重新开始训练。

#### Concat pooling
文本分类的信号通常包含在几个字中，这可能出现在文本中的任意位置。由于输入文档中可能包含数百个单词，如果仅仅考虑最后的隐藏装填，信息可能丢失。所以，我们对于最后一步的hidden state和所有hidden states的max pooling和均值pooling的结果
$$ H = \{h_1, ..., h_T\} $$
$$ h_c = [h_T, maxpool(H), meanpool(H)] $$

微调分类任务是迁移学习最关键的部分。过度调节可能导致灾难性的遗忘，消除通过预训练得到的信息。过于谨慎有可能导致收敛缓慢以及过拟合。

### Discriminative ﬁne-tuning

由于不同层能够捕捉到不同类型的信息，他们应该进行不同程度的调整。

适当处理不同层的最简单方法是一次一层的训练模型，类似于贪心层次训练方法和“链解冻”。然后这引入了一个顺序的要求，阻碍了并行性。并且**每次训练都会重新遍历一次数据集，很容易在小的数据集上过拟合。**所以，提出使用discriminative fine-tuning

与其对所有的层使用相同的学习率，有区别的微调允许我们使用不同的学习率调节每个层，对于上下文，在时间步t的参数更新公式为：
![](https://raw.githubusercontent.com/gjwei/gjwei.github.io/master/uploads/20180517144054.png)

对于discriminative fine-tuning,我们将参数按照层进行划分为$\{ \theta^1, ..., \theta^L\}$。相似的，也可以得到每一层的学习率

然后，在进行参数更新的时候，根据以下规则进行调整参数。
![](https://raw.githubusercontent.com/gjwei/images/master/20180517152238.png)

经验发现，首先选择第L层的学习率，然后根据$\eta^l = \eta^{l + 1} * 0.3$效果很好

### BPTT for Text Classiﬁcation (BPT3C)

语言模型通过反向传播进行更新参数，为了对大规模文档分类器进行精细调节，我们提出BPTT3C
1. 首先将文档划分大小b的批次
2. 在每个批次训练的开始，模型参数初始化为之前批次的最后的状态
3. 跟踪最大池化和平均池化的隐藏状态
4. 梯度被反向传播到哪些对预测有贡献的隐藏状态的参数

实际上，我们可以使用可变长的反向传播序列。

### Bidirectional LM

## Experiments
![](https://raw.githubusercontent.com/gjwei/images/master/20180517152156.png)


没有找打相关的代码，不知道具体的实现是怎么样的。
感觉这种方法还是主要使用与分类任务上。期待代码的公布和测试了。