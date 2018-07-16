---
title: mix-of-softmax
date: 2018-07-12 22:17:56
tags: NLP, Deep learning
---
# BREAKING THE SOFTMAX BOTTLENECK: A HIGH-RANK RNN LANGUAGE MODEL
ICLR2018


## Abstract
将LM看做是一个matrix分解的问题，现有的softmax-based models会受限于Softmax bottleneck。作者证明了，使用对于分布式的word embedding 使用softmax没有能力model language。
作者提出了MoS的方法，改进了softmax。

<!-- more -->

## Introduction
LM的问题可以归结为通过某个word的上下文context，预测next-token.
这是一个conditional probability问题。

通常的方法，使用rnn将context编码成一个vector，然后通过softmax求得条件概率，这种方法会很大程度的改变了context。

基于softmax的方法是没有足够能力去model language的。这就是softmax bottleneck。

作者提出Mixture of Softmaxes

![](https://raw.githubusercontent.com/gjwei/images/master/20180712222507.png)

作者的github有开源MoS的代码，但是是pytorch版本的，项目需要使用tf，所以就自己改写了，发现，运行速度上要比NCE慢了许多，对比来看，NCE对于训练的加速还是很明显的。训练结果来看，MoS会更好点。

tf实现的代码：

```python
# coding: utf-8
# Author: gjwei
import tensorflow as tf
from model.BiLM import ELMO
import ipdb

class ELMoMoS(ELMO):
    """reference: https://github.com/zihangdai/mos/blob/master/model.py
        paper: https://arxiv.org/abs/1711.03953.pdf
    """

    def __init__(self, config):
        super(ELMoMoS, self).__init__(config)


    def add_loss_op(self):
        # use mix of softmax
        with tf.variable_scope("loss"):

            mask = tf.sequence_mask(self.label_lengths)

            rnn_output = tf.concat((self.rnn_output[0][:, :-2, :], self.rnn_output[1][:, 2:, :]), axis=-1)
            # reshape it to [-1, self.rnn_output_dim]
            inputs = tf.reshape(rnn_output, shape=[-1, self.rnn_output_dim * 2])  # shape is [bs x step, nhidlast]

            # bs * step, nhidlast ——> bs * step, n_expers * ninp
            latent = tf.layers.dense(inputs, units=self.config.n_experts * self.config.ninp, activation=tf.nn.tanh,
                                     kernel_initializer=tf.glorot_uniform_initializer(), name='latent')
            # bs * step, n_experts * nipn --> bs * step * n_experts, ninp --> bs * step * n_experts, ntoken
            latent = tf.reshape(latent, shape=[-1, self.config.ninp])
            logit = tf.layers.dense(latent, units=self.config.ntokens,
                                    kernel_initializer=tf.glorot_uniform_initializer(), name='logit')

            # bs * step, nhidlast --> bs * step, n_experts
            prior_logit = tf.layers.dense(inputs, units=self.config.n_experts, use_bias=False,
                                          kernel_initializer=tf.glorot_uniform_initializer(), name='prior_logit')
            # prior_logit = tf.reshape(prior_logit, shape=[-1, self.config.n_experts])

            prior = tf.nn.softmax(prior_logit, axis=-1)  # bs * step, n_experts

            prop = tf.reshape(tf.nn.softmax(tf.reshape(logit,
                                                       shape=[-1, self.config.ntokens])),
                              shape=[-1, self.config.n_experts, self.config.ntokens])  # [bs * step, n_experts, ntokens]
            prior = tf.expand_dims(prior, axis=2)  # bs * step, n_experts, 1

            prop = tf.reduce_sum(tf.multiply(prop, prior), axis=1)  # bs * step * n_tokens

            self.logits = tf.reshape(prop, [-1, self.config.nwords])  # bs * step, n_tokens

            # labels = tf.reshape(self.labels, shape=[-1, 1])
            labels = tf.reshape(tf.one_hot(self.labels, depth=self.config.ntokens), shape=[-1, self.config.ntokens])
            losses = 1 * tf.losses.log_loss(labels=labels, predictions=self.logits, reduction=tf.losses.Reduction.NONE)


            # losses = tf.reshape(losses, shape=tf.shape(self.labels))
            # ipdb.set_trace()
            losses = tf.reshape(tf.reduce_sum(losses, axis=-1), shape=tf.shape(self.labels))
            losses = tf.boolean_mask(losses, mask)

            with tf.variable_scope("train"):
                # concat forward outputs and backward outputs

                # ipdb.set_trace()

                # print(self.rnn_output[0].get_shape)


                self.loss = tf.reduce_mean(losses)

                # for tensorboard
                tf.summary.scalar("train_loss", self.loss)
                # We have to use e instead of 2 as a base, because TensorFlow
                # measures the cross-entropy loss with the natural logarithm

            with tf.variable_scope("eval"):

                self.loss_eval = tf.reduce_mean(losses)

                self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                                           tf.int32)
                self.pred_and_target_logits = [tf.reduce_max(self.logits, axis=-1)] 

                # for tensorboard
                tf.summary.scalar('dev_loss', self.loss_eval)

```