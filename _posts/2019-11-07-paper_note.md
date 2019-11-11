---
layout: post
comments: true
title: inferential machine comprehension
date: 2019-11-07 15:32:10
tags: machine comprehension
---

> Meta-RL is meta-learning on reinforcement learning tasks. After trained over a distribution of tasks, the agent is able to solve a new task by developing a new RL algorithm with its internal activity dynamics. This post starts with the origin of meta-RL and then dives into three key components of meta-RL.

<!--more-->

# Inferential Machine Comprehension: Answering Questions by Recursively Deducing the Evidence Chain from Text

## career
- inferential machine comprehension

## problems
- exsisting methods may be good at context matching, but have limited capability to reason across multiple sentences to deduce the right answer.

## improvement
- mimic human process for deducing the evidence chain
- reinforcement learning

## difficulty
- design a iterative process while the task is rather simple: output the option with highest score

------

# Coherent Comments Generation for Chinese Articles with a Graph-to-Sequence Model

## career
- comment generation

## problems
- traditional LSTM is not good at generating a short review form a long document
- comment is more diverse compared with mt

## improvement
- propose a topic interaction graph to represent the article
- graph to sequence  generation

## difficulty
- data collection
- graph construction from raw text

----

# Adversarial Attention Modeling for Multi-dimensional Emotion Regression

## career
- emotion regression
- Adversarial Multi-task Learning 

## problem
-  multiple independent models for different emotion dimensions
-  relationship between two emotion dimensions can be leveraged

## improvement
- a multi-task learning task (shared information and task-specifical information)
- adversarial learning ensures that shared feature space simply contains task-invariant information

----

# Adversarial Multi-task Learning for Text Classification

## career
- multi-task learning

## problems
- divide the features of different tasks into private and shared spaces
- the shared feature space could contain some unnecessary taskspecific features, while some sharable features could also be mixed in private space, suffering from feature redundancy

## improvement
- adversarial multi-task
- orthogonality constraints

---

## Divide, Conquer and Combine: Hierarchical Feature Fusion Network with Local and Global Perspectives for Multimodal Affective Computing

## career
- Multimodal machine learning

## problems
- treats the feature vectors of the modalities as the smallest units and fuse them at holistic level
- variations across different portions of a feature vector which may contain disparate aspects of information,  thus fail to render the fusion procedure more specialized

## improvement
- leverage a sliding window to explore inter-modality dynamics locally
- global fusion to obtain an overall view of multimodal embeddings via a specifically designed ABS-LSTM
- integrate Regional Interdependence Attention and Global Interaction Attention

----
# OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs

## career
- dialog system
- specifically, retrieve entities to mention in the next turn

## improvement
- multi-modal feature representation
- zero-shot learning
- adversarial training
- domain adaption

---
# Domain-Adversarial Training of Neural Networks

## career
- domain adaptation

## improvement
- domain classifier
- gradient reversal layer
- discriminative for the main learning task on the source domain and indiscriminate with respect to the shift between the domains


![domain_adaption](/../assets/post_image/domain_adaption.jpg)

