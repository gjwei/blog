---
title: deeplearning笔记-chapter2
date: 2018-05-05 16:47:47
tags: deeplearning笔记
---
# 第二章：线性代数
## 概念
1. 标量
2. 向量
3. 矩阵：二维数组
4. 张量：超过二维的数组
5. 转置

在运算中，我们允许矩阵和向量相加，这种是隐式的复制向量b到很多位置，以满足相加的条件的方式，叫做**广播**

## 矩阵相乘
很了解，忽略不看了。

## 单位矩阵和逆矩阵

## 线性相关和生成子空间

子空间是指原始数据线性组合能达到的点的集合

## 特征分解
将矩阵分解成一组特征向量和特征值的形式。
特征向量定义为：
$$Av=\lambda v$$
标量$\lambda$称为特征值。
矩阵的特征分解可以记做：
$$A=Vdiag(\lambda) V^{-1}$$

正定矩阵：所有的特征值都是正的
负定矩阵：特征值都是负的
半正定矩阵：特征是都是非负的
半负定矩阵：特征值都是非正的

## 奇异值分解
$$ A=UDV^{T}$$
矩阵U，V都是正交矩阵，D是对角矩阵

我们可以从特征分解的角度理解SVD

A的左奇异值向量是$AA^T$的特征向量

A的右奇异值是$A^TA$的特征向量


