---
title: leaning-to-rank-for-IR
date: 2019-01-23 19:02:01
tags: Learning to rank
---
# Learning to Rank for Information Retrieval
https://www.cda.cn/uploadfile/image/20151220/20151220115436_46293.pdf
## 目标：本周内读完前11页的内容

## Chapter 1 Overiew
本章主要是介绍l2r，首先以文档搜索为例介绍排名问题，然后，回顾传统方法，提出广泛使用的评估方法，之后给出机器学习解决排名问题的冬季，并对现有的方法进行分类和简化描述

### 介绍
![](https://raw.githubusercontent.com/gjwei/images/master/20190123190626.png)
搜索引擎包括：爬虫，解析器，索引器，链接分析器，查询处理器和排名起。
爬虫复杂爬去网页信息；解析器复杂分析文档并生成对应的索引和超链接图。
ranking在搜索引擎中是处于核心作用，在其他信息检索任务中也是同样的呃核心作用，如协同过滤，问答，多媒体检索，文本摘要等。

### 1.2 Ranking in IR
## 传统的方法
**相关性排名模型**
相关性排名模型根据文档和查询之间的相关性产生排序的结果。通常将每个单独的文档作为输入，并计算匹配的分之，然后按照这些分值进行降序排序。
早期相关性的排名模型基于文档中查询项的出现来检索文档，例子包括布尔模型。基本上，这些模型可以预测文档中是否与查询相关，但是无法预测相关的程度。
为了进一步建立相关度模型，提出了**向量空间模型**，文档和查询都表示为欧几里得空间中的向量，向量的内积可以表示为它们的相关性。其中TF-IDF常用语表示查询和文档的向量。

虽然VSM暗示了查询单词之间的独立性，隐藏语义索引LSI避免了这种假设，尤其是使用SVD作为讲原始线性空间分解为“潜在语义空间”。然后使用这个新空间中的相似性来定义查询和文档之间的相似性。

另外，基于概率排序模型也取得很大成功。BM2和语言模型等著名的排序模型都可以看作是概率排序模型。BM25是按照文档相关性的对数概率对文档进行排序。
给定一个query q，包含了很多的单词t1,t2,...,tM, 计算一个文档的BM25值的公式：
![](https://raw.githubusercontent.com/gjwei/images/master/20190124155407.png)
其中TF表示文档d中单词t的出现频率，LEN(d)表示文档d的长度（字数），avdl表示文档集合的平均文档长度。$k_1$和$b$表示自由参数，控制每个项的权重。
LMIR是将统计LM应用到IR上。统计LM会对一个序列的单词分配一个改了。当应用到IR上，LM会与文档相关联。利用查询query q，基于文档会产生query的可能性$P(q|d)=\prod_{i=1}^{M}P(t_i|d)$.
![](https://raw.githubusercontent.com/gjwei/images/master/20190124160106.png)
$\lambda$表示平滑系数。


#### 重要的排序模型
PageRank模型
PageRank使用随机点击链接的浏览者到达特定网页对网页进行排名的概率。 在一般情况下，任何页面的PageRank值都可以表示为
![](https://raw.githubusercontent.com/gjwei/images/master/20190124160303.png)
网页$d_u$的PageRank值依赖于所有指向du的网页的rank值处于该网页的外链数。

#### Query-Level Position-Based Evaluations 基于query位置的评估方法
有了模型，我们就需要一个可以用来衡量模型好坏的评估方法。

Cranfield 实验方法：
* 收集很多随机选择的query
* 对于每个query
  * 收集与这个query相关的文档集$\{d_j\}_{J=1}^M$/
  * 人工对这些文档进行相关度评价
  * 使用模型对文档集进行排序
  * 使用评估方法衡量模型排序结果和真实结果的差异
* 使用所有query评估结果的平均值作为模型的性能结果

相关度判断方法：
1. 相关程度：人工标注判断一个文档是否相关。假设对同一个qeury q的两个文档，我们分配对应的相关度量。如果两个文档$d_u$和$d_v$，对应的相关度$l_u>L_V$那么，u就要比v更相关。
2. 成对偏好：指定一个文档比其他文档更相关。如果一个文档u要比文档v更相关，我们标定$l_{u,v}=1$,否则就标定为-1.
3. 整体顺序：对于给定的query标定文档集的整体排序。对于文档集合$\{d_j\}_{j=1}^M$，将其表示为一个特定的序列，表示$\pi_l$

**大多数评估测量是首先针对每个查询定义的，作为排名模型和相关性判断给出的排序列表π的函数。然后对测试集中的所有查询求平均度量。**
一下介绍的方法都是通过最大化值进行优化。如MRR，MAP，NDCG等，我们可以通过考虑使用1减去这些值来最小化值作为优化目标。

**Mean Reciprocal Rank（MRR)** 平均倒数排名
对于query q，排序结果中第一个相关文档的位置表示为$r_1$, MRR表示为r1的倒数。

![](https://raw.githubusercontent.com/gjwei/images/master/20190124164556.png)

**Mean Average Precision**MAP
首先，我们定义位置k(P@K)的精确度。假设我们有两个标签相关和不想关。1表示相关，0表示不想关。
![](https://raw.githubusercontent.com/gjwei/images/master/20190124164750.png)
![](https://raw.githubusercontent.com/gjwei/images/master/20190124164821.png)
然后，就可以定义AP：
![](https://raw.githubusercontent.com/gjwei/images/master/20190124164935.png)
m表示和query q相关联的所有文档的数目。$m_1$表示为相关文档的树木。
对于上图1.4，我们可以计算得到：P@1=1,p@2=1/2,p@3=2/3,然后，AP=5/6

**Discounted Cumulative Gain (DCG)**
DCG可以根据多个有序列类别来利用相关性判断，并在定义中具有明确的折扣因子。假设查询q的排序列表是pi，那么位置k的DCG表示为：
![](https://raw.githubusercontent.com/gjwei/images/master/20190124170122.png)
在这里，G(.)表示文档的评级，通常设置为$G(z)=(2^z-1)$ $\eta(.)$表示为位置的折扣因子,通常设置为$\eta(j)=1/log(j+1)$。

对其进行归一化，除去DCG@k的最大值。得到归一化的NDCG:
![](https://raw.githubusercontent.com/gjwei/images/master/20190124170444.png)

以下摘自链接：http://sofasofa.io/forum_main_post.php?postid=1002561

![](https://raw.githubusercontent.com/gjwei/images/master/20190124171041.png)

**Rank Correlation (RC)**

总结下：
1. 所有评价方法都是计算在query level的。
2. 都是基于位置进行度量的。这些方法通常是不可微的，所以优化起来很麻烦。


## Learning to rank

### Machine Learning Framework
许多机器学习研究中，一下几个关键组成部分：
1. 输入空间。通常，对象是由特征向量组成，根据不同的应用进行提取。
2. 输出空间，包含了针对输入对象的学习目标。机器学习的输出空间有两个相关但是不同的定义。首先是任务的输出空间，她高度依赖于应用的场景。第二个是让学习过程更容易的输出空间，这个可能和任务的输出不同。例如，使用回归方法解决分类任务的时候。
3. 假设空间。定义了输入空间到输出空间的函数类。
4. 为了优化参数，通常使用训练集。损失函数测量模型输出和真实标签的一致程度。利用损失函数，可以在训练集上定义经验损失，通过经验损失最小化学习最优假设。
![](https://raw.githubusercontent.com/gjwei/images/master/20190124201042.png)
### Learning to rank定义
我们定义使用机器学习完成排序任务的方法为learning-to-rank方法。
多数的算法都是通过监督学习优化query-documnet对的特征。我们称具有以下两种属性的方法为leanring-to-rank方法
1. Feature Based。
基于特征意味着所有的文档都是用向量表示。学习排名的典型特征包括文档中查询术语的频率，BM25模型和PageRank模型的输出，甚至本文档与其他文档之间的关系。
2. 判别训练
判别训练是指leanring to rank有自己的输入空间，输出空间，假设空间和损失函数。
判别式训练事是从训练数据中自动学习的过程，这个对于真实搜索引擎是十分重要的，因为搜索引擎每天都会接受大量用户反馈和使用log，自动学习并不断改进排名机制是非常重要的。

### Learning-to-Rank Framework
![](https://raw.githubusercontent.com/gjwei/images/master/20190124202318.png)

排名算法分为三种：PointWise，pairwise，listwise
不同的方法，定义不同的输入和输出空间，使用不同的假设函数以及不同的损失函数。


## Pointwise方法
输入空间是包含单个文档的特征向量。
输出空间是每个文档的相关度，可以根据相关程度，转换为不同的标签。ground truth label表示正确的标注。
* 如果判断是直接通过相关程度$l_j$给出，那么文档的标签就可以定义为$y_j=l_j$
* 如果判断是给定了一个成对的偏好$l_{uv}$，则可以通过计算击败其他文档的频率来得到文档的正确标签。
* 如果判断结果是给定了一个总顺序$\pi_l$。则可以通过使用映射函数获得真是标签。例如，文档在$\pi_l$的位置可以当作是真实标签。

假设空间包含了将文档的特征向量作为输入并预测文档的相关度的函数，我们称之为**评分函数**。基于评分函数，我们可以对文档进行排序生产最终的排序列表。

损失函数检查了每个文档的真实标签预测准确度，在不同的pointwise算法中，排序是作为一个回归，分类模型来做的，因此相应的损失函数也可以当作该算法的损失函数

需要注意的是，**逐点方法**不考虑文档之间的相互依赖性，因此文档在最终的排序结果的位置对于其损失函数是不可见的，该方法没有利用某些文档实际上与同一个查询相关联的事实。这种发发有其局限性。

### Pairwise方法
输入空间是成对的文档，都表示为特征向量。
输出空间也是包含了成对的属性（我们从{+1,-1}中取值）。根据成对偏好，我们可以将不同类型的判断转化为标签。
* 如果相关程度定义为$l_j$，那么$(x_u,x_v)$成对标签就可以定义为$y_{u,v}=2*I_{l_u>l_v}-1$
* 如果相关程度是直接给定的偏好，就直接设定$y_{u,v}=l_{u,v}$
* 如果给定的是总的顺序$\pi_l$，定义$y_{u,v}=2*I_{\pi_l(u)<\pi_l(v)}-1$

假设空间包含了双变量函数h，他将一对文档作为输入并输出它们的相对顺序。为了简单，我们使用评分函数f定义，即：$h（x_u，x_v）= 2·I \{f（x_u）> f（x_v）\}  -  1$。

损失函数是衡量$h(x_u,x_v)$和真实标签$y_{u,v}$。

成对方法使用的损失函数仅仅考虑了两个文档之间的相对顺序。然而，仅仅查看一对文档，很难得出最终的列表的位置。考虑到信息检索中都是使用query级别和位置，我们可以看到这种方法和信息检索的排序之间的差别。

#### Listwise方法
输入空间是一个queyr对应的相关联的文档集合,如$x=\{x_j\}_{j=1}^m$
输出空间是文档的排序列表。不同的判断发过誓可以转化为排序列表的标签。
* 对于给定了相关度的判别，按照相关度的大小排序得到一个列表作为真实标签。
* 如果给定的是pairwise的偏好，同样定义一个满足所有pairwise偏好的列表座位真实标签。
* 给定一个总的列表，直接定义标签为$\pi_y=\pi_l$

对于列表方法，学习过程是为了让输出空间与任务的输出完全相同。优化过程更直接。

假设空间包含了多变量的函数h，会对一个文档集进行处理并预测他们的序列。假设函数h通常由评分函数f来实现，例如$h(x) = sort \odot f(x)$

Listwise方法的损失函数通常有两种，一种是和评价方法有关（度量特定损失函数），另外一种无关（非度量特定损失函数）。我们主要参考一下标准来将listwise方法和其他两种方法区分：
* 列表损失函数是根据一个query以及对应的所有训练文档定义的
* liswise损失函数不能完全分解为单个文档或者文档对的求和
* 强调排序列表的概念，并且最终的排序结果的文档位置在损失函数中是可见的。
  

总结表：
![](https://raw.githubusercontent.com/gjwei/images/master/20190125123408.png)

文章使用的符号：
![](https://raw.githubusercontent.com/gjwei/images/master/20190125123701.png)

# Chapter2 The Pointwise方法
本章讨论了基于回归算法，基于分类的算法和基于序数的肃反啊，然后讨论他们的优缺点。
对于基于回归的算法，输出空间包括实数值相关分数。对于分类的算法=，输出空间包含了无序的类别，对于基于序数的算法，输出空间输出有序的类别。

## Regression-Based Algorithms
排序问题被当作一个回归问题。

### Subset Ranking with Regression
给定了 query q对应的文档集$x=\{x_j\}_{j=1}^m$,真实标签 $y=\{y_j\}_{j=1}^m$. 打分函数f用来排序这些文档。损失函数定义为：

![](https://raw.githubusercontent.com/gjwei/images/master/20190125124943.png)

只有当评分函数 能够准确输出时候，没有损失，否则就会有二次的损失。对于相关文档，只有当打分函数准确输出为1时候，损失为0，否则如果打分为2，看起来是对这个文档具有更强的相关预测，也会有损失。所以，某种意义上，这样的损失函数不合理。



作为一种拓展，有提出一种加权回归模型。权重有助于模型更集中雨相关文档的回归错误。另外，还对回归损失增加了正则化项，减少过拟合风险。

平方损失的理论是基于NDCG的排名误差的理论上限，然而根据以上的讨论，即使有较大的损失，对应的排序结果可能也是最优的。我们可以认为平方损失是基于NDCG排名错误的松散约束。



## Classiﬁcation-Based Algorithms

#### 二元分类

##### 基于SVM的方法

假设文档对应的标签y为1表示相关，-1表示不想关

![image-20190125144519164](/Users/gjwei/Library/Application Support/typora-user-images/image-20190125144519164.png)

其实就是1以文档的特征向量作为输入，使用SVM做二元分类判断该文档是否和query匹配。



##### Logistic Regression-Based Method

![](https://raw.githubusercontent.com/gjwei/images/master/20190125144758.png)

### Multi-class Classiﬁcation for Ranking

#### Boosting Tree-Based Method

给定了一个文档集合 $x=\{x_j\}^m_{j=1}$.对应的相关度为 $y={y_j}_{j=1}^m$. 假设又一个多元分类器，预测文档对应的相关度。

然后，用于学习分类器的损失函数是定义为0-1分类错误的替代函数：

![](https://raw.githubusercontent.com/gjwei/images/master/20190125145158.png)

在增强树算法，我们使用以下代理损失函数：

![image-20190125145407636](/Users/gjwei/Library/Application Support/typora-user-images/image-20190125145407636.png)

#### Association Rule Mining-Based Method 基于关联规则挖掘的方法

从训练数据中发现文档归属于某一类的特征。



### 基于序数回归的算法

此时，输出空间包含的是有序类别。

通常是找到一个打分函数，然后用一系列阈值对得分进行分割，得到有序类别。采用 PRanking、基于 margin 的方法都可以。

###### Perceptron-Based Ranking (PRanking)

Pranking的目的是在投影文档之后找到参数向量定义的方向，在文档上可以容易使用阈值将文档区分为不同的有序类别。

##### Ranking with Large Margin Principles

主要是使用SVM技术学习模型参数和阈值b

## 讨论

### 与相关反馈的关系

学习排名的逐点方法，特别是基于分类的算法，与相关反馈算法有很强的相关性

相关反馈算法在信息检索中有重要作用。基本思想是从显示，隐式或者盲目的反馈中学习以更新原始的query，新的query用于检索得到一组新文档，通过迭代式执行这个操作，我们可以使query更接近于最优query，从而提高检索性能。

如Rocchio算法：

![image-20190125152124659](/Users/gjwei/Library/Application Support/typora-user-images/image-20190125152124659.png)

所以，我们相当于将query vector看作是一个模型的参数。

这种方法和learning 2 rank方法的区别在于；

1. 该算法空间是VSM算法空间，query和document都表示为向量，他们之间的内积当作是相关程度。与此不同的是，l2r算法中，特征空间是从每一个query-document对中抽取得到的。
2. 该算法从反馈中学习模型的参数，然后去对和相同query的文档进行新的排序。l2r则是会对未知的query进行排序验证
3. 该算法的模型参数w实际上具有物理意义，即是更新的查询向量。而l2r则没有这个含义，并且仅对应于每个特征对于排名任务的重要性。
4. 算法的目的是更新query以获得更好的结果，但是不学习最佳的排名。



### Pointwise算法问题

1. pointwise 类方法并没有考虑同一个 query 对应的 docs 间的内部依赖性。一方面，导致输入空间内的样本不是 IID 的，违反了 ML 的基本假设，另一方面，没有充分利用这种样本间的结构性。其次，当不同 query 对应不同数量的 docs 时，整体 loss 将会被对应 docs 数量大的 query 组所支配，前面说过应该每组 query 都是等价的。

2. ranking 追求的是排序结果，并不要求精确打分，只要有相对打分即可。
3. 损失函数也没有 model 到预测排序中的位置信息。因此，损失函数可能无意的过多强调那些不重要的 docs，即那些排序在后面对用户体验影响小的 doc。

### 改进方法

RankCosine介绍了一种**query-level normalization factor**，定义的损失函数是基于query q的得分函数f和真实标签的得分向量的cosine相似度。

![](https://raw.githubusercontent.com/gjwei/images/master/20190125153246.png)

# Chapter 3 The PairWise 方法

Pairwise方法不关注与每个文档的相关程度，它只关心与两个文档之间的相对顺序，从这个意义讲，它更接近于“排名”的概念。



在成对方法汇总，排名通常被简化为文档对的分类，即，确定一对文档中哪个文档是优选的。也就是，学习的目标是最小化错误分类文档的数量。



## 示例算法

