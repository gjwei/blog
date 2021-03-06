---
layout: recommendations
title: Filtering
date: 2019-01-21 10:29:19
tags: recommend system
---
# Recommendations Item-to-Item Collaborative Filtering

推荐系统是使用消费者的兴趣作为输入生产一个推荐项目的列表。很多的应用只有使用顾客购买的item以及明确的评价，但是他们可能也会有其他的属性，包括查看的项目，人口统计数据，主题兴趣和喜好的艺术家等。

点击量：广告被点击的次数
点击率：点击率：广告被点击的比例，点击量/展现量
转化次数：访问这到达转换目标页的次数
转化率（VR）：转换次数/访问次数

零售商拥有大量数据，数千万的用户和商品
1. 许多场景需要在半秒内返回结果，同时要生产高质量的建议
2. 新客户冷启动问题
3. 老客户可以获取稻粱信息
4. 数据不稳定：每次交互都能提供有价值的客户数据，算法必须要即使响应新的信息

## Recommendation Algorithms
多数的推荐系统都是首先找到一组和当前客户的购买和评价的商品有重叠的用户，算法聚集这些相类似的顾客，消除该用户已经购买的商品，然后推荐给这个用户剩余的商品。

### 传统的协同过滤算法
传统方法将客户表示为项目的N维度向量，其中N是不同的项目目录。向量的组成部分对于购买或者评级为正的项目为正，对于负评级项目为负。
为了抵消流行商品的影响，算法会通过乘以一个反频率（购买或者评价商品顾客数量的倒数），使不知名的商品更具有相关性。对于所有用户来说，这个向量是十分稀疏的。
该算法基于与用户最相似的少数客户生成推荐，他可以以各种方式衡量两个顾客A和B的相似度，可以通过余弦距离测量。
通过对一个商品计算有多少相似顾客会购买它来进行排序。

使用协同过滤的方法复杂度很高，最差会有O(MN)的复杂度。M是顾客数据，而N是每个顾客对应的N个商品。

## 聚类模型
使用一个相似度衡量方法，聚类算法可以将最相似的几个顾客聚到一起组成一个族。然后根据同一个Cluster中的顾客作为相似顾客做推荐

## Search-Based方法
基于搜索，也叫做基于内容的推荐方法，是通过搜索相关商品来做推荐。
给定一个用户的购买记录，算法能够构造出一个搜素的query，去找到其他相关的相似商品。


## Item-to-item协同过滤算法
amazon会提供购物车建议，基于他们的购物车推荐给用户商品。amazon使用推荐算法为每个客户的兴趣进行个性化，item-to-item协同过滤，可以实时生成高质量的建议。

### 如何工作？
这种方法并不是将客户和类似的客户做匹配，而是将客户购买的商品的每一个和类似的商品做匹配，然后将这些商品组合到推荐列表中。
为了确定给定商品的最相似匹配，该算法通过查找客户倾向于一起购买的商品构建项目表。
![](https://raw.githubusercontent.com/gjwei/images/master/20190121152912.png)

这个方法的优势在于，可以在离线状态下计算similar-items，然后线上能够实时推荐用户相似的商品。
而传统的方法无法保证这一点。