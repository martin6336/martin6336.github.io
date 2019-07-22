---
layout: post
comments: true
title: ELMo
date: 2019-07-22 15:32:10
tags: summarization reinforcement-learning
---

> Meta-RL is meta-learning on reinforcement learning tasks. After trained over a distribution of tasks, the agent is able to solve a new task by developing a new RL algorithm with its internal activity dynamics. This post starts with the origin of meta-RL and then dives into three key components of meta-RL.

<!--more-->

# ELMo(Embeddings from Language Models)

## challenges

- complex characteristics of word use(e.g., syntax and semantics)
- how these uses vary across linguistic contexts 

## model structure

### multi-layers bi-lstm

the fundamental of this model is a multi-layers bi-lstm

### ELMo

ELMo is a task specific combination of the intermediate layer representations in the biLM


$$
\begin{aligned} R_{k} &=\left\{\mathbf{x}_{k}^{L M}, {\mathbf{h}}_{k, j}^{LM_{fw}}, 
{\mathbf{h}}_{k, j}^{LM_{bw}} | j=1, \ldots, L\right\} \\ &=\left\{\mathbf{h}_{k, j}^{L M} | j=0, \ldots, L\right\} \end{aligned}
$$

ELMo collapses all layers in R into a single vector, can simply select the top layers or use a weighted sum function. $$\mathbf{E} \mathbf{L} \mathbf{M} \mathbf{o}_{k}^{t a s k}$$ is the representation of the k-th word.

$$
\mathbf{E} \mathbf{L} \mathbf{M} \mathbf{o}_{k}^{t a s k}=E\left(R_{k} ; \Theta^{t a s k}\right)
$$

### apply the ELMo in supervised tasks

Once pretrained, the biLM can compute representations for any task. The input of the RNN is $$\left[\mathbf{x}_{k} ; \mathbf{E} \mathbf{L} \mathbf{M} \mathbf{o}_{k}^{t a s k}\right]$$ ,further improvement can be achieved by replacing $$\mathbf{h}_k$$ with $$\left[\mathbf{h}_{k} ; \mathbf{E} \mathbf{L} \mathbf{M} \mathbf{o}_{k}^{t a s k}\right]$$.



In some circumstances, fine tuning the biLM on domain specific data leads to significant drops in perplexity and an increase in downstream task performance.