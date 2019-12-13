---
layout: post
comments: true
title: paper note
date: 2019-11-07 15:32:10
tags: machine comprehension
---

> paper's main idea,  domain of the problem, and their innovate methods 

<!--more-->

# Inferential Machine Comprehension: Answering Questions by Recursively Deducing the Evidence Chain from Text

## career
- inferential machine comprehension

## problems
- existing methods may be good at context matching, but have limited capability to reason across multiple sentences to deduce the right answer.

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
- a multi-task learning task (shared information and task-specific information)
- adversarial learning ensures that shared feature space simply contains task-invariant information

----

# Adversarial Multi-task Learning for Text Classification

## career
- multi-task learning

## problems
- divide the features of different tasks into private and shared spaces
![domain_adaption](/../assets/post_image/ad_text.jpg)
- the shared feature space could contain some unnecessary task specific features, while some sharable features could also be mixed in private space, suffering from feature redundancy

## improvement
- adversarial multi-task
- orthogonality constraints
![domain_adaption](/../assets/post_image/ad_text1.jpg)

---

# Divide, Conquer and Combine: Hierarchical Feature Fusion Network with Local and Global Perspectives for Multimodal Affective Computing

## career
- Multi-modal machine learning

## problems
- treats the feature vectors of the modalities as the smallest units and fuse them at holistic level
- variations across different portions of a feature vector which may contain disparate aspects of information,  thus fail to render the fusion procedure more specialized

## improvement
- leverage a sliding window to explore inter-modality dynamics locally
- global fusion to obtain an overall view of multi-modal embeddings via a specifically designed ABS-LSTM
- integrate Regional Interdependence Attention and Global Interaction Attention

----
# OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs

## career
- dialog system
- specifically, retrieve entities to mention in the next turn

## improvement
- multi-modal feature representation
- zero-shot learning(domain adaption)
- adversarial training

---
# Domain-Adversarial Training of Neural Networks

## career
- domain adaptation

## improvement
- domain classifier
- gradient reversal layer
- discriminative for the main learning task on the source domain and indiscriminate with respect to the shift between the domains


![domain_adaption](/../assets/post_image/domain_adaption.jpg)

----



# Joint Slot Filling and Intent Detection via Capsule Neural Networks 

## career
- NLU
- slot filling
- intent detection
- multi-task

## problems
- existing works treat slot filling and intent detection separately in a pipeline manner or adopt joint models which sequentially label slots while summarizing the utterancelevel intent
- the hierarchical relationship among words, slots, and intents has not been sufficiently used

## improvement
- the properties of **Capsule Neural Networks** are appealing for dealing with the hierarchical structure: word, slot, intent

---

# Semi-supervised Domain Adaptation for Dependency Parsing(水)

## career
- Dependency Parsing
- domain adaptation
- semi-supervised

## problems
- cross-domain parsing assuming there is no target-domain training data in an unsupervised manner
- the annotation cost is high

## improvement
- semi-supervised domain-adaption can be tackled with only dozens or hundreds of labled target-domain traning sentences

---

# DEEP BIAFFINE ATTENTION FOR NEURAL DEPENDENCY PARSING

## career
- dependency parsing

## problems
- whether there exists an arc between two words
- what is the label of the arc

## improvement
- DEEP BIAFFINE ATTENTION
- variable-class bi-affine classifier for arc 
![bi-affine](/../assets/post_image/bi_affine.jpg)
- fixed-class bi-affine classifier for label

---

# This Email Could Save Your Life: Introducing the Task of Email Subject Line Generation

## career
- summarization

## improvement
- multi-stage training strategy
- Supervised Pretraining
- RL Training for Extractor(not novel)
![email](/../assets/post_image/email.jpg)
- new dataset

----

# Incremental Learning from Scratch for Task-Oriented Dialogue Systems

## career
- dialogue system

## problems
- unexpected queries may be given to the system after the system is deployed. 

## improvement
- new dataset
- Incremental Dialogue System consists of three main components: dialogue embedding module, uncertainty estimation module and online learning module.
![incre_dialog](/../assets/post_image/incre_dialog.jpg)
- variation as confidnece(**probabilistic graphical models**)
- Monte Carlo approximation


# Multi-Modal Sarcasm Detection in Twitter with Hierarchical Fusion Model

## career
- multi-modal learning

## improvement
- text, attribute, image embedding matrix
- use text, attribute, image **guidance vectors** to get three **modality feature** vectors
- two layer feed-forward neural network to calculate modalities' weight
![sarcasm](/../assets/post_image/multi-modal-sarcasm.jpg)

# Using Human Attention to Extract Keyphrase from Microblog Post

## career
- Keyphrase extraction

## improvement
- estimates human attention from GECO corpus
- use the TRT feature, which represents total human attention on words during reading

# Topic-Aware Neural Keyphrase Generation for Social Media Language

## career
- keyphrase extraction

## base
- Discovering Discrete Latent Topics with Neural Variational Inference


## improvement
- neural topic model
![topic_keyphrase.jpg](/../assets/post_image/topic_keyphrase.jpg)

# Topic Memory Networks for Short Text Classification

## improvement
- neural topic model
- memory network
![topic_memory](/../assets/post_image/topic_memory.jpg)


# Generating Summaries with Topic Templates and Structured Convolutional Decoders

## career
- summarization

## improvement
- topic guidance (auxiliary task)


# Self-Supervised Learning for Contextualized Extractive Summarization

## career
- summarization

## improvement
- self-supervised
- mask, replace, switch

# Token-level Dynamic Self-Attention Network for Multi-Passage Reading Comprehension

## career
- reading comprehension

## improvement
- structure innovation
- use estimated importance to extract the most important K tokens. 
- lower memory consumption and makes the self-attention focus on the active part of a long input sequence
- ![topic_memory](/../assets/post_image/dynamic_san.jpg)

# PaperRobot: Incremental Draft Generation of Scientific Ideas

## career
- paper abstract, conclusion, and future work generation

## improvement
- memory, graph entity
- combine decoder, memory, language model to predict
- ![topic_memory](/../assets/post_image/paperRobot.jpg)
