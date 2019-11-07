---
layout: post
comments: true
title: inferential machine comprehension
date: 2019-11-07 15:32:10
tags: machine comprehension
---

> Meta-RL is meta-learning on reinforcement learning tasks. After trained over a distribution of tasks, the agent is able to solve a new task by developing a new RL algorithm with its internal activity dynamics. This post starts with the origin of meta-RL and then dives into three key components of meta-RL.

<!--more-->

# Sentence Centrality Revisited for Unsupervised Summarization

## challenges

- it is unrealistic to expect that large-scale and high-quality training data will be available  under different circumstances

## improvement
- employ BERT to capture semantic feature and capture sentence similarity
- edges should be directed

## directed edges
direted edges can be implemented in an easy yet effective way:

$$
centrality\left(s_{i}\right)=\lambda_{1} \sum_{j<i} e_{i j}+\lambda_{2} \sum_{j>i} e_{i j}
$$

$$ \lambda_{1}, \lambda_{2} $$ are different weights for forwardand backward-looking directed edges

## BERT as sentence encoder
we fine-tune BERT based on a type of sentence-level distributional hypothesis. We use the idea of negative sampling inseated of skip-thought vectors. The objective aims to distinguish context sentences ($$s_{i-1}, s_{i+1}$$)from other sentences.

## similarity matrix
employ pair-wise dot product