---
layout: 'acl2018:chinese'
title: acl2018 Chinese NER Using Lattice(网格)LSTM
date: 2018-05-11 14:12:58
tags: NLP, paper
---

## 摘要部分
论文采用一种网格结构的LSTM模型，用来做汉语的NER，这个模型能够编码一串输入的字（characters），同时还有所有可能的words，用来匹配一个字典。
<!-- more -->
这种方法相比于基于字的方法，可以更好地利用wrod和word sequence 的信息。

相比于基于word的方法，Lattice LSTM不会遇到分割错误的问题。

门控的RNN可以使得模型能够选择最相关的characters和words，从而得到更好的编码信息。实验也表明了这种Lattice LSTM效果更好。

## 介绍（introduction)
目前，英文的NER的state-of-the-art模型是LSTM-CRF，character的信息也有正和到word的表示中。

Chinese NER和分词密切相关。命名实体的边界就是word的边界。一种直觉的解决方法是，先分词，在做NER。

然后，这种方法会有问题，如果分词不对，将会导致NER的错误。分词问题还没有解决，所以，这种方法可能会导致很严重的问题。

character-based NER模型没有使用word和sequence的信息。所以，我们使用网格结构的LSTM表示sentence中的word的信息，将其整合到character-based的LSTM-CRF中。

由于格子中有指数数量的字-字路径，我们利用lattice lstm自动控制sequence从开始到结束的信息流。
![](https://raw.githubusercontent.com/gjwei/images/master/20180511144703.png)
门控的cells用来动态的路由到每个字符。通过对NER数据的训练，lattice LSTM可以从上下文宗找到更多有用单词，从而获得更好的性能。
和character sequence labeling任务相比，模型有用到显示的word信息，而且不会遇到分词带来的错误信息。

Lattice可以看做是tree-structed的RNN的拓展。

## 模型介绍
我们采用LSTM-CRF作为模型的主要结构。
使用的符号介绍如下：
一个输入sequence可以表示为：
$$ s = (c_1, ..., c_m) $$
$c_j$表示为第j个character
也可以表示为：
$$ s = (w_1, ..., w_n) $$
$w_i$表示第i个word
$t(i, k)$用来表示第i个word中的第k个character

### Character-Based model
![](https://raw.githubusercontent.com/gjwei/images/master/20180511153418.png)
每个character都可以表示为其embedding表示
$$ x_j^c=e^c(c_j) $$
$e^c$表示为a character embedding lookup table.

一个标准的CRF模型使用hidden states$h^c$作为输入进行sequence labeling

* char+bichar

通过concat bigram的embedding结果，增强字符的表示。
$$ x_j^c = [e^c(c_j); e^b(c_j, c_{j + 1})] $$

* char+softword

使用分割得到的word作为character的一个增强信息使用。
$$ x_j^c = [e^c(c_j); e^s(seg(c_j))] $$

$e^s$用来表示分割label的embedding lookup table。

### word-based model
$$ e_i^w = e^w(w_i) $$

#### 整合character的表达
1. word+char LSTM
通过双向的LSTM学习单词$w_i$的hidden states。
$h_{t(i,1)}^c, ..., h_{t(i, len(w))}$
然后就可以得到word $w$中i-th的character的表达
![](https://raw.githubusercontent.com/gjwei/images/master/20180511155807.png)
2. word + char CNN
![](https://raw.githubusercontent.com/gjwei/images/master/20180511155928.png)

### Lattice model
![](https://raw.githubusercontent.com/gjwei/images/master/20180511160009.png)

模型中总共有四种vectors
`input vector, output hidden vecotr, cell vector, gate vector`
使用recurrent 模型得到一个character cell vector $c_j^c$和一个hidden vector $h_j^c$。在这里$c_j^c$用来表示recurrent信息的流动。
$h_j$用于CRF。

LSTM原理

![](https://ws1.sinaimg.cn/large/9244e6f1gy1fr7i3mfe0zj20fo06sdg7.jpg)

和单纯的character-based model相比，$c_j^c$考虑到了词典的subsequents$w_{b,e}^d$，每个sebsequence可以表示为：
$$ X_{b, e}^w = e^w(w_{b, e}^d) $$

问题：如何才能使用到word的信息呢？
通过LSTM，可以得到word$c_{b,e}^w$对应的context vector.
![](https://ws1.sinaimg.cn/large/9244e6f1gy1fr7ie247hdj20gb06vwf1.jpg)

通过subsequence 的表达 $c_{b, e}^w$ ，可以讲更多的信息流入到$c_e^c$

距离而言，在图二中的`长江大桥`，因为字典中存在两个word`长江，长江大桥`。我们会将所有的word表示$c_{b, e}^w$连接到cell $c_e^c$上。

得到了word的表达$c_{b, e}^w$之后，可以使用一个门用来控制每个单词对最终得到的表达的影响。

![](https://ws1.sinaimg.cn/large/9244e6f1gy1fr7iw2z1h1j20dk030t8o.jpg)

进而可以总和所有的word的信息和我们得到的character的信息，得到最终的cell state
![](https://ws1.sinaimg.cn/large/9244e6f1gy1fr7ix3s2fkj20gp0c53zz.jpg)

将最终得到的h输入到CRF中。

### Decoding and Training
一个标准的CRF层是使用顶层的h作为输入。
对于Lattice model，输入为$h_1, h_2, .., h_n$
因为有n个characters

![](https://ws1.sinaimg.cn/large/9244e6f1gy1fr8b5qnbupj20my07zdgo.jpg)

在这里$y'$表示任意的label sequence，$W_{CRF}^{l_i}$表示$h_i$对于label $l_i$的状态特征的参数。
$b_{CRF}^{(l_{i-1},l_i)}$表示bias，用来表示状态转移特征的参数，只依赖于当前位置的label和上一个label

使用Viterbi算法进行求解。

损失函数定义为：

![](https://ws1.sinaimg.cn/large/9244e6f1gy1fr8bkgxgkcj20hg034jrg.jpg)

## 实验以及实验结果
![](https://ws1.sinaimg.cn/large/9244e6f1gy1fr8bmrv3itj20iu0ovq7q.jpg)