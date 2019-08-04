---
layout: post
comments: true
title: summarization
date: 2019-07-04 15:32:10
tags: summarization reinforcement-learning
---

> Meta-RL is meta-learning on reinforcement learning tasks. After trained over a distribution of tasks, the agent is able to solve a new task by developing a new RL algorithm with its internal activity dynamics. This post starts with the origin of meta-RL and then dives into three key components of meta-RL.

<!--more-->

# summarization

## 1. Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond(2016 coNLL)

### problems to solve
- 和MT类似，但是seq2seq的两头没有一一对应关系
- 对信息的压缩 in a lossy manner，MT是loss-less

### contribution
- attentional encoder-decoder RNN
- propose novel models
- propose a new dataset

### model structure
**1**. Encoder-Decoder RNN with Attention
基本模型依据[Bahdanau](https://arxiv.org/pdf/1409.0473v7.pdf "Neural machine translation by jointly learning to align and translate")提出的模型。

- encoder consists of a bidirectional GRU-RNN
- decoder consists of a uni-directional GRU-RNN($$s_{i}=f(s_{i-1}, y_{i-1}, c_{i})$$)
- attention mechanism over the source-hidden states
- a soft-max layer over target vocabulary to generate words
- large vocabulary ‘trick’ -LVT(target vocab+source document vocab in a batch)

**2**. Capturing Keywords using Feature-rich Encoder.
encoder的输入为word embedding+NER tags+TF+IDF+POS

**3**. Modeling Rare/Unseen Words using Switching Generator-Pointer
起源于LVT，每次生成计算switch，为1则使用target vocab，为0使用pointer从原文复制单词。switch概率公式如下：

$$
\begin{aligned} P\left(s_{i}=1\right) &=\sigma\left(\mathbf{v}^{s} \cdot\left(\mathbf{W}_{h}^{s} \mathbf{h}_{i}+\mathbf{W}_{e}^{s} \mathbf{E}\left[o_{i-1}\right]\right.\right.\\ &+\mathbf{W}_{c}^{s} \mathbf{c}_{i}+\mathbf{b}^{s} ) ) \end{aligned}
$$

指向各个单词的概率为，i为decoder中位置，j为encoder中位置：

$$
\begin{aligned} P_{i}^{a}(j) & \propto \exp \left(\mathbf{v}^{a} \cdot\left(\mathbf{W}_{h}^{a} \mathbf{h}_{i-1}+\mathbf{W}_{e}^{a} \mathbf{E}\left[o_{i-1}\right]\right.\right.\\ &+\mathbf{W}_{c}^{a} \mathbf{h}_{j}^{d}+\mathbf{b}^{a} ) ) \end{aligned}
$$

模型的目标函数为，$$g_i$$是switch的真实值，只在训练阶段才有：

$$
\begin{array}{l}{\log P(\mathbf{y} | \mathbf{x})=\sum_{i}\left(g_{i} \log \left\{P\left(y_{i} | \mathbf{y}_{-i}, \mathbf{x}\right) P\left(s_{i}\right)\right\}\right.} \\ {+\left(1-g_{i}\right) \log \left\{P\left(p(i) | \mathbf{y}_{-i}, \mathbf{x}\right)\left(1-P\left(s_{i}\right)\right)\right\} )}\end{array}
$$

模型结构如下，计算switch时缺少attention连线

![en_de](/../assets/post_image/en_de.png)

**4**.  Capturing Hierarchical Document Structure with Hierarchical Attention
计算context vector $$c_t$$时使用hierarchical attention，对上文的attention进一步处理而已

$$
P^{a}(j)=\frac{P_{w}^{a}(j) P_{s}^{a}(s(j))}{\sum_{k=1}^{N_{d}} P_{w}^{a}(k) P_{s}^{a}(s(k))}
$$


## 2. Abstractive Document Summarization with a Graph-Based Attentional Neural Model(2017 ACL)

### problems to solve
- key factor of summarization: saliency, fluency, coherence, and novelty
- neural generation are naturally good at fluency, saliency has not been addressed
- generate long sequences; same sentences or phrases are often repeated in the output

### contribution
- discover the salient information of a document
- a novel graph-based attention mechanism
- a hierarchical decoding algorithm with a reference mechanism

### model structure
The left denotes the traditional Bahdanauet's attention [$$s_{i}=f(s_{i-1}, y_{i-1}, c_{i})$$], while the right half denotes the graph-based attention.
![graph-based attention](/../assets/post_image/graph-based attention.png)

1. hierarchical encoder-decoder framework

$$\alpha_i^j$$ indicates how much the $i$-th original sentence $$s_i$$contributes to generating the $$j$$-th sentence in summary. $$c_j$$ is the context vector.

$$
   \mathbf{c}_{j}=\sum_{i} \alpha_{i}^{j} \mathbf{h}_{i}
$$

$$
   \alpha_{i}^{j}=\frac{e^{\eta\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{\prime}\right)}}{\sum_{l} e^{\eta\left(\mathbf{h}_{l},\mathbf{h}_{j}^{\prime}\right)}}
$$

2. graph-based attention

   Traditional attention methods not good at judging which sentences are more important to a document. *A sentence is important if it's heavily linked with many important sentences according to Pagerank algorithm.* $$W(i,j) = h_i^TMh_j$$ is the adjacent matrix. $$\mathbf{y} \in \mathcal{R}^{n}$$ with all elements equal to $$\frac{1}{n}$$. The importance score of each sentence is :

$$
   \mathbf{f}=(1-\lambda)\left(I-\lambda W D^{-1}\right)^{-1} \mathbf{y}
$$

Further, we want compute the rank scores of the original sentences regarding $$h_j$$. Applying the idea of topic-sensitive pagerank, which alters $$\mathbf{y}$$.

$$
\mathbf{y}_{T}=\left\{\begin{array}{ll}{\frac{1}{|T|}} & {i \in T} \\ {0} & {i \notin T}\end{array}\right.
$$

In order to penalize the model from attending to previously attended sentences. The graph-based attention is finally computed as:

$$
\alpha_{i}^{j}=\frac{\max \left(f_{i}^{j}-f_{i}^{j-1}, 0\right)}{\sum_{l}\left(\max \left(f_{l}^{j}-f_{l}^{j-1}, 0\right)\right)}
$$

3. model training
loss function:

$$
   \mathcal{L}=\sum_{(Y, X) \in \mathcal{D}}-\log p(Y | X ; \theta)
$$

4. hierarchical decoding algorithm
   

	A beam search strategy may help to alleviate the repetition in a sentence, but the **repetition** in the whole generated summary is remained a problem. Sentence-level beam search is realized by maximizing the accumulated score of all the sentences generated.

   We add an additional term to the score $$\tilde{p}\left(y_{\tau}\right)$$, ref is a function calculates the ratio of bigram overlap between two texts.

$$
score(\tilde{y_{\tau}}) = \tilde{p}\left(y_{\tau}\right)+
\gamma\left(\operatorname{ref}\left(Y_{\tau-1}+y_{\tau}, s_{*}\right)-\operatorname{ref}\left(Y_{\tau-1}, s_{*}\right)\right)
$$


## 3. A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION(2018 ICLR)
### problem
- repeating phrase problem
- exposure bias
- a large number of summaries are potentially valid, which MLL objective doesn't take this flexibility into account

### contribution

- intra-temporal attention in encoder-decoder
- new objective function combining the maximum-likelihood-entropy loss with rl objective

![deep_reinforce](/../assets/post_image/deep_reinforce.png)

### model structure

3 key element: $$c_t^e, c_t^d, h_t^d$$

$$c_t^e$$ is the inpute context vectore, $$c_t^d$$ is the decoder context decoder, $$h_t^d$$ is the hidden state vector of decoder at time step $$t$$.


- 
$$
f\left(h_{t}^{d}, h_{i}^{e}\right)=h_{t}^{d^{T}} W_{\mathrm{attn}}^{e} h_{i}^{e}
$$

$$
   e_{t i}=f\left(h_{t}^{d}, h_{i}^{e}\right)
$$

define new temporal attention score, penalizing input tokens that have obtained high attention scores in past attention steps.

$$
e_{t i}^{\prime}=\left\{\begin{array}{ll}{\exp \left(e_{t i}\right)} & {\text { if } t=1} \\ {\frac{\exp \left(e_{t i}\right)}{\sum_{j=1}^{t-1} \exp \left(e_{j i}\right)}} & {\text { otherwise }}\end{array}\right.
$$

$$
\alpha_{t i}^{e}=\frac{e_{t i}^{\prime}}{\sum_{j=1}^{n} e_{t j}^{\prime}}
$$

obtain final context vector, a weighted sum of all hidden vector in encoder. It differs at different decoding step $$t$$.

$$
c_{t}^{e}=\sum_{i=1}^{n} \alpha_{t i}^{e} h_{i}^{e}
$$


- 
similar as the way we obtain $$c_t^e$$. To further avoid repeating, we can incorporate more information about the previously decoded sequence into the decoder.

$$
e_{t t^{\prime}}^{d}=h_{t}^{d^{T}} W_{\mathrm{atn}}^{d} h_{t^{\prime}}^{d}
$$

$$
\alpha_{t t^{\prime}}^{d}=\frac{\exp \left(e_{t t^{\prime}}^{d}\right)}{\sum_{j=1}^{t-1} \exp \left(e_{t j}^{d}\right)}
$$

$$
c_{t}^{d}=\sum_{j=1}^{t-1} \alpha_{t j}^{d} h_{j}^{d}
$$

- adopt switch function $$u_t$$ in generation process. 

$$
p\left(u_{t}=1\right)=\sigma\left(W_{u}\left[h_{t}^{d}\left\|c_{t}^{e}\right\| c_{t}^{d}\right]+b_{u}\right)
$$

same as pointer network, tradition generation probability is:

$$
p\left(y_{t} | u_{t}=0\right)=\operatorname{softmax}\left(W_{\mathrm{out}}\left[h_{t}^{d}\left\|c_{t}^{e}\right\| c_{t}^{d}\right]+b_{\mathrm{out}}\right)
$$

the copy probability is defined as:

$$
p\left(y_{t}=x_{i} | u_{t}=1\right)=\alpha_{t i}^{e}
$$

### objective
combine two objective, $$\hat{y}$$ is the baseline output maximizing the output probability distribution at each time step, $$y^s$$ is obtained by sampling from the probability distribution at each time step $$t$$(the whole sentence?), $$y^{*}$$ is ground truth, $$r(y)$$ is the reward function, 

$$
L_{m l}=-\sum_{t=1}^{n^{\prime}} \log p\left(y_{t}^{*} | y_{1}^{*}, \ldots, y_{t-1}^{*}, x\right)
$$

$$
L_{r l}=\left(r(\hat{y})-r\left(y^{s}\right)\right) \sum_{t=1}^{n^{\prime}} \log p\left(y_{t}^{s} | y_{1}^{s}, \ldots, y_{t-1}^{s}, x\right)
$$

$$
L_{\text {mixed}}=\gamma L_{r l}+(1-\gamma) L_{m l}
$$

### trick
- ground-truth summaries almost never contain the same trigram twice
- introduce some weight-sharing between this embedding matrix and the $$W_{out}$$ matrix of the token generation layer

## 4. Improving Abstraction in Text Summarization(2018 EMNLP)
### problem
- word overlap metrics do not capture the abstractive nature
### contribution
- decouples the **extraction** and **generation** responsibilities of the decoder by factoring it into a **contextual network** and a **language model**.
- a mixed objective that jointly optimizes the **n-gram overlap** with the ground-truth summary while encouraging **abstraction**.

### model structure

#### ![improve_rl](/../assets/post_image/improve_rl.png)

#### contextual network the same as above

#### language model

The input of language model is the same as decoder at each time step $$t$$. Combing $$h^{lm}_t$$ with  $$[h_{t}^{d}|c_{t}^{e}| c_{t}^{d}]$$, we have $$h^{fuse}_t$$

$$
\begin{aligned} f_{t} &=\operatorname{sigmoid}\left(W^{\operatorname{lm}}\left[r_{t} ; h_{3, t}^{\operatorname{lm}}\right]+b^{\operatorname{lm}}\right) \\ g_{t} &=W^{\mathrm{fuse}}\left(\left[r_{t} ; g_{t} \odot h_{3, t}^{\operatorname{lm}}\right]\right)+b^{\mathrm{fuse}} \\ h_{t}^{\mathrm{fuse}} &=\operatorname{ReLU}\left(g_{t}\right) \end{aligned}
$$

#### abstractive reward

changing the the reward function  $$r(y)$$.



## 5. SummaRuNNer(2017 AAAI)

### problem

### contribution

### model structure

![SummaRuNNer](/../assets/post_image/SummaRuNNer.png)

### extractive training

The first layer of the RNN runs at the word level, second layer of bi-directional RNN that runs at the sentence-level. Using average pooling to get sentence or document representation.

$$
\mathbf{d}=\tanh \left(W_{d} \frac{1}{N_{d}} \sum_{j=1}^{N^{d}}\left[\mathbf{h}_{j}^{f}, \mathbf{h}_{j}^{b}\right]+\mathbf{b}\right)
$$

The binary decision is defined as below. Each row represents content, salience, novelty, absolute and relative positional embedding respectively.

$$
\begin{array}{r}{P\left(y_{j}=1 | \mathbf{h}_{j}, \mathbf{s}_{j}, \mathbf{d}\right)=\sigma\left(W_{c} \mathbf{h}_{j}\right.} \\ {+\mathbf{h}_{j}^{T} W_{s} \mathbf{d}} \\ {-\mathbf{h}_{j}^{T} W_{r} \tanh \left(\mathbf{s}_{\mathbf{j}}\right)} \\ {+W_{a p} \mathbf{p}_{j}^{r}} \\ {+W_{r p} \mathbf{p}_{j}^{r}} \\ {+b )}\end{array}
$$

$$
\mathbf{s}_{j}=\sum_{i=1}^{j-1} \mathbf{h}_{i} P\left(y_{i}=1 | \mathbf{h}_{i}, \mathbf{s}_{i}, \mathbf{d}\right)
$$

The objective is to minimize the negative log-likelihood of the observed labels at training time:

$$
\begin{aligned} l(\mathbf{W}, \mathbf{b}) &=-\sum_{d=1}^{N} \sum_{j=1}^{N_{d}}\left(y_{j}^{d} \log P\left(y_{j}^{d}=1 | \mathbf{h}_{j}^{d}, \mathbf{s}_{j}^{d}, \mathbf{d}_{d}\right)\right.\\ &+\left(1-y_{j}^{d}\right) \log \left(1-P\left(y_{j}^{d}=1 | \mathbf{h}_{j}^{d}, \mathbf{s}_{j}^{d}, \mathbf{d}_{d}\right)\right) \end{aligned}
$$

### how to get extractive label

We employ a greedy approach, where we add one sentence at a time incrementally to the summary, such that the Rouge score of the current set of selected sentences is maximized with respect to the entire summary.

### abstractive training

We need to label sentences in the document for extractive training, which will bring loss. This paper propose a novel training technique to train SummaRuNNer abstractively. We **couple the initial model with a RNN decoder** that models the generation of abstractive summaries at training time only. We modify the first three equations of GRU based RNN, $$s_{-1}$$ is the summary representation computed at the last sentence of the sentence-level bidirectional RNN. It's the only bridge between the initial model and the coupled decoder.


$$
\begin{aligned} \mathbf{u}_{j} &=\sigma\left(\mathbf{W}_{u x} \mathbf{x}_{j}+\mathbf{W}_{u h} \mathbf{h}_{j-1}+\mathbf{b}_{u}\right) \\ \mathbf{r}_{j} &=\sigma\left(\mathbf{W}_{r x} \mathbf{x}_{j}+\mathbf{W}_{r h} \mathbf{h}_{j-1}+\mathbf{b}_{r}\right) \\ \mathbf{h}_{j}^{\prime} &=\tanh \left(\mathbf{W}_{h x} \mathbf{x}_{j}+\mathbf{W}_{h h}\left(\mathbf{r}_{j} \odot \mathbf{h}_{j-1}\right)+\mathbf{b}_{h}\right) \\ \mathbf{h}_{j} &=\left(1-\mathbf{u}_{j}\right) \odot \mathbf{h}_{j}^{\prime}+\mathbf{u}_{j} \odot \mathbf{h}_{j-1} \end{aligned}
$$

$$
\begin{array}{ll}{\mathbf{u}_{k}=} & {\sigma\left(\mathbf{W}_{u x}^{\prime} \mathbf{x}_{k}+\mathbf{W}_{u h}^{\prime} \mathbf{h}_{k-1}+\mathbf{W}_{u c}^{\prime} \mathbf{s}_{-1}+\mathbf{b}_{u}^{\prime}\right)} \\ {\mathbf{r}_{k}=} & {\sigma\left(\mathbf{W}^{\prime} r x_{k}+\mathbf{W}_{r h}^{\prime} \mathbf{h}_{k-1}+\mathbf{W}_{r c}^{\prime} \mathbf{s}_{-1}+\mathbf{b}_{r}^{\prime}\right)} \\ {\mathbf{h}_{k}^{\prime}=} & {\tanh \left(\mathbf{W}_{h x}^{\prime} \mathbf{x}_{k}+\mathbf{W}_{h h}^{\prime}\left(\mathbf{r}_{k} \odot \mathbf{h}_{k-1}\right)+\right.}{\mathbf{W}_{h c}^{\prime} \mathbf{s}_{-1}+\mathbf{b}_{h}^{\prime} )}\end{array}
$$

The emission at each time-step is determined by a feed-forward layer $f$ followed by a softmax layer that assigns $\mathbf{P}_{k}$

$$
\begin{aligned} \mathbf{f}_{k} &=\tanh \left(\mathbf{W}_{f h}^{\prime} \mathbf{h}_{k}+\mathbf{W}_{f x}^{\prime} \mathbf{x}_{k}+\mathbf{W}^{\prime} f_{c} \mathbf{s}_{-1}+\mathbf{b}_{f}^{\prime}\right) \\ \mathbf{P}_{\mathbf{v}}(\mathbf{w})_{k} &=\operatorname{softmax}\left(\mathbf{W}_{v}^{\prime} \mathbf{f}_{k}+\mathbf{b}_{v}^{\prime}\right) \end{aligned}
$$

we minimize the negative log-likelihood of the words in the reference summary as follows.
$$
l\left(\mathbf{W}, \mathbf{b}, \mathbf{W}^{\prime}, \mathbf{b}^{\prime}\right)=-\sum_{k=1}^{N_{s}} \log \left(\mathbf{P}_{\mathbf{v}}\left(w_{k}\right)\right)
$$
where $$N_s$$ is the number of words in the reference summary.


## 6.Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting(2018 ACL)

### problem

- abstractive model suffers slow and inaccurate encoding of very long documents
- abstractive model suffers from repetitions
- a two-stage extractive-abstractive architecture is non-differentiable

### contribution

- a **extractive-abstractive architecture** , in which a extractor agent select salient sentence first, then an abstractive net to rewrite the sentence into summary
- apply actor-critic policy gradient to train extractor
- parallel decoding bring speed-up

###  extractive model structure

![sum_fast](/../assets/post_image/sum_fast.png)

- temporal convolutional model
- a bidirectional LSTM
- added LSTM(decoder)

temporal convolutional model compute a coarse sentence representation $$r_j$$; the follow bi-LSTM learning a stronger representation $$h_j$$ by taking into account all previous and future sentences. 



- sentence selection
using a pointer networker(decoder) to extract sentences. The decoder performs a 2-hop attention mechanism

$$
\begin{aligned} a_{j}^{t} &=v_{g}^{\top} \tanh \left(W_{g 1} h_{j}+W_{g 2} z_{t}\right) \\ \alpha^{t} &=\operatorname{softmax}\left(a^{t}\right) \\ e_{t} &=\sum_{j} \alpha_{j}^{t} W_{g 1} h_{j} \end{aligned}
$$

$$
u_{j}^{t}=\left\{\begin{array}{cc}{v_{p}^{\top} \tanh \left(W_{p 1} h_{j}+W_{p 2} e_{t}\right)} & {\text { if } j_{t} \neq j_{k}} \\ {} & {\forall k<t} \\ {-\infty} & {\text { otherwise }}\end{array}\right.
$$

$$
P\left(j_{t} | j_{1}, \ldots, j_{t-1}\right)=\operatorname{softmax}\left(u^{t}\right)
$$

where $$z_t$$ is the ouput of the added lstm

- get the extraction label for each sentence
we find the most similar document sentences $$d_{j_t}$$ as below. Given the proxy training labels, the extraction is then trained to minimize the cross-entropy loss.

$$
j_{t}=\operatorname{argmax}_{i}\left(\operatorname{ROUGE-L}_{r e c a l l}\left(d_{i}, s_{t}\right)\right)
$$

### abstractive net

Given the training paris(pairing each summary sentence with its extracted document sentence), the network is trained as usual seq2seq model to minimize the cross-entropy loss.

### reinforce-guide extraction

![sum_fast_rl](/../assets/post_image/sum_fast_rl.png)

At each time step $$t$$, the agent observe the current state $$c_t=(D, d_{j_{t-1}})$$, sample an action $$j_t \sim P(j_{t} | j_{1}, \ldots, j_{t-1})$$ to extract a sentence $$d_{j_{t}}$$ and receive a reward:

$$
r(t+1)=\operatorname{ROUGE-L}_{F_{1}}\left(g\left(d_{j_{t}}\right), s_{t}\right)
$$

Instead of using policy gradient algorithm, we apply a **advantage actor-critic(A2C) network**. We denote the parameters of extractor agent by $$\theta=\left\{\theta_{a}, \omega\right\}$$ for the decoder and hierarchical encoder respectively.

$$
A^{\pi_{\theta}}(c, j)=Q^{\pi_{\theta_{a}, \omega}}(c, j)-V^{\pi_{\theta_{a}, \omega}}(c)
$$

$$
\begin{array}{c}{\nabla_{\theta_{a}, \omega} J\left(\theta_{a}, \omega\right)=}  {\mathbb{E}\left[\nabla_{\theta_{a}, \omega} \log \pi_{\theta}(c, j) A^{\pi_{\theta}}(c, j)\right]}\end{array}
$$

The pointer net treats **EOE** as one of the extraction candidates and hence naturally results in a stop action in the stochastic policy.

### repetition-avoiding reranking

We keep all $$k$$ sentence candidates generated by beam search, rerank all $$k^n$$ combinations of the $$n$$ generated summary sentences. The rank score is the **number of repeated N-grams**. This paper also apply a [diverse decoding algorithm](https://arxiv.org/pdf/1611.08562.pdf)  favoring choosing hypotheses from diverse parents.

![sum_fast_diverse](/../assets/post_image/sum_fast_diverse.png)

## 7.Guiding Generation for Abstractive Text Summarization based on Key Information Guide Network(2018 NAACL)
### problem

- models are hard to be controlled in the process of generation without guidance, which leads to a lack of **key information**

### contribution
- combine the extractive model and abstractive model, using the former one to obtain keywords as guidance for the latter one
- key information guide network, which encode the keywords and integrates it into the abstractive model
- novel prediction-guide mechanism

### model structure

![sum_key](/../assets/post_image/sum_key.png)
#### the base encoder-decoder
the base model is similar with base pointer network

$$
\begin{aligned} e_{t i} &=v^{T} \tanh \left(W_{h} h_{i}+W_{s} s_{t}\right) \\ \alpha_{t}^{e} &=\operatorname{softmax}\left(e_{t}\right) \\ c_{t} &=\sum_{i=1}^{N} \alpha_{t i}^{e} h_{i} \end{aligned}
$$

$$c_t$$ is the context vector, which represents what has been read from the source text.

$$
P\left(y_{t} | y_{1}, \ldots, y_{t-1}\right)=\operatorname{softmax}\left(f\left(s_{t}, c_{t}\right)\right)
$$

#### use key information to improve model
extract keywords from the text by using **TextRank** algorithm, feed the keywords one-by-one into the key information network(bi-lstm). We concatenate $$\overline{h}_{1}$ and $\vec{h}_{1}$$ as the key information representation.

$$
k=\left[\begin{array}{c}{\overline{h}_{1}} \\ {\vec{h}_{n}}\end{array}\right]
$$

For the traditional generation part, changing the above equation into:
$$
e_{t i}=v^{T} \tanh \left(W_{h} h_{i}+W_{s} s_{t}+W_{k} k\right)
$$

$$
P_{v}\left(y_{t} | y_{1}, \ldots, y_{t-1}\right)=\operatorname{softmax}\left(f\left(s_{t}, c_{t}, k\right)\right)
$$

For the pointer net, a switch $$p_{sw}$$ is needed.

$$
p_{s w}=\sigma\left(w_{k}^{T} k+w_{c}^{T} c_{t}+w_{s_{t}}^{T} s_{t}+b_{s w}\right)
$$
The final probability distribution to predict the next word:

$$
\begin{aligned} P\left(y_{t}=w\right) &=p_{s w} P_{v}\left(y_{t}=w\right) +\left(1-p_{s w}\right) \sum_{i : w_{i}=w} \alpha_{t i}^{e} \end{aligned}
$$

The overall loss is:

$$
L=-\frac{1}{T} \sum_{t=0}^{T} \log P\left(y_{t}^{*} | y_{1}^{*}, \ldots, y_{t-1}^{*}, x\right)
$$

#### prediction-guide mechanism at test time

[prediction-guide mechanism](http://www.andrew.cmu.edu/user/hanqing1/data/vnn4nmt.pdf) is based on pairwise ranking loss. It's a single-layer feed forward network with sigmoid activation function. Compute the score of two partial summaries $$y_{p1}$$ and $$y_{p2}$$.
$$
\operatorname{Avg} \operatorname{Cos}\left(x, y_{p}\right)=\frac{1}{M} \sum_{\overline{s} \in S\left(y_{p}\right)} \cos (\overline{s}, k)
$$

$$
\overline{s}_{t}=\frac{1}{t} \sum_{l=1}^{t} s_{l}
$$

we hope the predicted value of $$v(s,y_{p1})$$ can be larger that $$v(s,y_{p1})$$ if $$AvgCos(x,y_{p1})>AvgCos(x,y_{p2})$$. So the loss function of the prediction-guide network is as follows:

$$
L_{p g}=\sum_{\left(x, y_{p 1}, y_{p 2}\right)} e^{v\left(x, y_{p 2}\right)-v\left(x, y_{p 1}\right)}
$$

where $$AvgCos(x,y_{p1})>AvgCos(x,y_{p2})$$.

At test time, we first compute the normalized log probability of each candidate, and then linearly combine it with the value predicted by the prediction-guide network. $$x=\left\{x_{1}, x_{2}, \dots, x_{N}\right\}$$ is the summary sequence. The final prediction probability is defined as:

$$
\alpha \times \log P(y | x)+(1-\alpha) \times \log v(x, y)
$$


## 8. Hierarchical Structured Self-Attentive Model for Extractive Document Summarization (HSSAS)(2018 IEEE)

### problem

- carrying the semantics along all-time steps of a recurrent model is relatively hard and not necessary

- the summary lies only on vector space that can hardly capture multi-topical content

- how to mirror the hierarchical structure of the document
- how to extract the most important sentences

### contribution(SummaRuNNer)
- propose a new hierarchical structured self-attention architecture

### model structure

![hssas](/../assets/post_image/hssas.png)

$$H_{s} \in \mathbb{R}^{n \times 2 u}$$ denotes the whole LSTM hidden states for a sentence, attention is calculated as below, then the sentence representation is $$s_{i}=a_{s}H_{s}$$

$$
a_{s}=\operatorname{softmax}\left(w_{s_{2}} \tanh \left(w_{s_{1}} H_{s}^{T}\right)\right)
$$

The operation is similar at sentence level.

Similar with SummaRunner, this model introduce sentence content richness $$C_j$$, salience with respect to the document $$M_j$$, the novelty of the sentence with respect to the accumulated summary $$N_j$$ and the positional feature $$P_j$$. $$o_j$$ is the summary representation at sentence $$j$$

$$
C_{j}=W_{c} s_{j}
$$

$$
M_{j}=s_{j}^{T} W_{s} d
$$

$$
N_{j}=s_{j}^{T} W_{r} \tanh \left(o_{j}\right)
$$

$$
o_{j}=\Sigma_{i=1}^{j-1} s_{i} P\left(y_{i}=1 | s_{i}, o_{i}, d\right)
$$

$$
P_{j}=W_{p} p_{j}
$$

The probability distribution for the sentence label $$y_j$$ is :

$$
P\left(y_{j}=1 | s_{j}, o_{j}, d\right)=\sigma\left(C_{j}+M_{j}-N_{j}+P_{j}+b\right)
$$

Training objective is defined as:

$$
\begin{aligned} l(W, b)=-& \Sigma_{d=1}^{N} \Sigma_{j=1}^{n_{d}}\left(y_{j}^{d} \log P\left(y_{j}^{d}=1 | s_{j}, o_{j}, d_{d}\right)\right.\\ &+\left(1-y_{j}^{d}\right) \log \left(1-P\left(y_{j}^{d}=1 | s_{j}, o_{j}, d_{d}\right)\right) \end{aligned}
$$





## 9. SWAP-NET: Sentences and Words from Alternating Pointer Networks(2018 ACL)
### problems
- sentence label are human-crafted features

### contribution
- design an architecture utilizes key words in the selection process
- Salient sentences of a document often contain key words


### modle structure

![swap_1](/../assets/post_image/swap_1.png) 

![swap_2](/../assets/post_image/swap_2.png)

The word embedding $$x_i$$ is encoded as $$e_i$$. The vector output of bi-lstm at the end of a sentence is used to represent the entire sentence. The hidden vector of word-level and sentence-level decoder are defined as:

$$
\begin{array}{l}{h_{j}=L S T M\left(h_{j-1}, a_{j-1}, \phi\left(A_{j-1}\right)\right)} \\ {H_{j}=L S T M\left(H_{j-1}, A_{j-1}, \phi\left(a_{j-1}\right)\right)}\end{array}
$$

where $$a_j=\sum_{i=0}^{n} \alpha_{ij}^{w} e_{i}$$, $$A_j=\sum_{k=0}^{N} \alpha_{kj}^{s} E_{k}$$.
The probability to select sentence or words is defined below:

$$
\alpha_{k j}^{s}=p\left(T_{j}=k | v_{<j}, Q_{j}=1, D\right)
$$

$$
\alpha_{k j}^{s}=\operatorname{softmax}\left(V_{T}^{T} \phi\left(W_{H} H_{j}+W_{T} E_{k}\right)\right)
$$

$$
\alpha_{i j}^{w}=p\left(t_{j}=i | v_{<j}, Q_{j}=0, D\right)
$$

$$
\alpha_{i j}^{w}=\operatorname{softmax}\left(v_{t}^{T} \phi\left(w_{h} h_{j}+w_{t} e_{i}\right)\right)
$$

The switch probability at $$j^{th}$$ decoding step is given by:

$$
\begin{array}{l}{p\left(Q_{j}=1 | v_{<j}, D\right)=} {\sigma\left(w_{Q}^{T}\left(H_{j-1}, A_{j-1}, \phi\left(h_{j-1}, a_{j-1}\right)\right)\right)} \\ {p\left(Q_{j}=0 | v_{<j}, D\right)=1-p\left(Q_{j}=1 | v_{<j}, D\right)}\end{array}
$$

The final action $$v_j$$ is defined as:

$$
\begin{aligned} p_{k j}^{s} &=\alpha_{k j}^{s} p\left(Q_{j}=1 | v_{<j}, D\right) \\ p_{i j}^{w} &=\alpha_{i j}^{w} p\left(Q_{j}=0 | v_{<j}, D\right) \end{aligned}
$$

$$
v_{j}=\left\{\begin{array}{ll}{k=\arg \max _{k} p_{k j}^{s}} & {\text { if } \max _{k} p_{k j}^{s}>\max _{i} p_{i j}^{w}} \\ {i=\arg \max _{i} p_{i j}^{w}} & {\text { if } \max _{i} p_{i j}^{w}>\max _{k} p_{k j}^{s}}\end{array}\right.
$$



The loss function $$l_j$$ at $$j^{th}$$ step is set to $$l_{j}=-\log \left(p_{k j}^{s} q_{j}^{s}+p_{i j}^{w} q_{j}^{w}\right)-\log p\left(Q_{j} | v_{<j}, D\right)$$, $$q^s_j$$ and $$q_j^w$$ is mutual exclusive. There is no ground-truth label.

The final summary consists of three sentences with the highest importance scores.

$$
I\left(s_{k}\right)=\alpha_{k j}^{s}+\lambda \sum_{w_{i} \in s_{k}} \alpha_{i l}^{w}
$$

## 10. Learning to Extract Coherent Summary via Deep Reinforcement Learning(2018 AAAI)

### problem 
- ignore the coherence of summary when extract sentences
- coherence is difficult to included into the objective function

### contribution
- 

### model structure

![coherence](/../assets/post_image/coherence.png)
