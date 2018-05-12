---
title: kmeans和em算法
date: 2018-05-03 22:01:10
tags: machine learning
---
# K-mean与高斯混合模型：K-means算法和EM算法的差别在哪里？
答：CSDN博主JpHu说，K-Means算法对数据点的聚类进行了“硬分配”，即每个数据点只属于唯一的聚类；而GMM的EM解法则基于后验概率分布，对数据点进行“软分配”，即每个单独的高斯模型对数据聚类都有贡献，不过贡献值有大有小。