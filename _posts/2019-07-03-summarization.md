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
- decoder consists of a uni-directional GRU-RNN($s_{i}=f(s_{i-1}, y_{i-1}, c_{i})$)
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

模型的目标函数为，$g_i$是switch的真实值，只在训练阶段才有：
$$
\begin{array}{l}{\log P(\mathbf{y} | \mathbf{x})=\sum_{i}\left(g_{i} \log \left\{P\left(y_{i} | \mathbf{y}_{-i}, \mathbf{x}\right) P\left(s_{i}\right)\right\}\right.} \\ {+\left(1-g_{i}\right) \log \left\{P\left(p(i) | \mathbf{y}_{-i}, \mathbf{x}\right)\left(1-P\left(s_{i}\right)\right)\right\} )}\end{array}
$$
模型结构如下，计算switch时缺少attention连线
![模型结构](D:\allkind\blog\martin6336.github.io\assets\post_image\en_de.png)

**4**.  Capturing Hierarchical Document Structure with Hierarchical Attention
计算context vector $c_t$时使用hierarchical attention，对上文的attention进一步处理而已
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
The left denotes the traditional Bahdanauet's attention [$s_{i}=f(s_{i-1}, y_{i-1}, c_{i})$], while the right half denotes the graph-based attention.
![graph-based attention](D:\allkind\blog\martin6336.github.io\assets\post_image\graph-based attention.png)

1. hierarchical encoder-decoder framework

$\alpha_i^j$ indicates how much the $i$-th original sentence $s_i$ contributes to generating the $j$-th sentence in summary. $c_j$ is the context vector.
$$
   \mathbf{c}_{j}=\sum_{i} \alpha_{i}^{j} \mathbf{h}_{i}
$$

$$
   \alpha_{i}^{j}=\frac{e^{\eta\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{\prime}\right)}}{\sum_{l} e^{\eta\left(\mathbf{h}_{l},\mathbf{h}_{j}^{\prime}\right)}}
$$

2. graph-based attention

   Traditional attention methods not good at judging which sentences are more important to a document. *A sentence is important if it's heavily linked with many important sentences according to Pagerank algorithm.* $W(i,j) = h_i^TMh_j$ is the adjacent matrix. $\mathbf{y} \in \mathcal{R}^{n}$ with all elements equal to $\frac{1}{n}$. The importance score of each sentence is :
$$
   \mathbf{f}=(1-\lambda)\left(I-\lambda W D^{-1}\right)^{-1} \mathbf{y}
$$

Further, we want compute the rank scores of the original sentences regarding $h_j$. Applying the idea of topic-sensitive pagerank, which alters $\mathbf{y}$.

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

   We add an additional term to the score $\tilde{p}\left(y_{\tau}\right)$, ref is a function calculates the ratio of bigram overlap between two texts.
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

![deep_reinforce](D:\allkind\blog\martin6336.github.io\assets\post_image\deep_reinforce.png)

### model structure

3 key element: $c_t^e, c_t^d, h_t^d$

$c_t^e$ is the inpute context vectore, $c_t^d$ is the decoder context decoder, $h_t^d$ is the hidden state vector of decoder at time step $t$.


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

obtain final context vector, a weighted sum of all hidden vector in encoder. It differs at different decoding step $t$.

$$
c_{t}^{e}=\sum_{i=1}^{n} \alpha_{t i}^{e} h_{i}^{e}
$$


- 
similar as the way we obtain $c_t^e$. To further avoid repeating, we can incorporate more information about the previously decoded sequence into the decoder.

$$
e_{t t^{\prime}}^{d}=h_{t}^{d^{T}} W_{\mathrm{atn}}^{d} h_{t^{\prime}}^{d}
$$

$$
\alpha_{t t^{\prime}}^{d}=\frac{\exp \left(e_{t t^{\prime}}^{d}\right)}{\sum_{j=1}^{t-1} \exp \left(e_{t j}^{d}\right)}
$$

$$
c_{t}^{d}=\sum_{j=1}^{t-1} \alpha_{t j}^{d} h_{j}^{d}
$$

- adopt switch function $u_t$ in generation process. 

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
combine two objective, $\hat{y}$ is the baseline output maximizing the output probability distribution at each time step, $y^s$ is obtained by sampling from the probability distribution at each time step $t$(the whole sentence?), $y^{*}$ is ground truth, $r(y)$ is the reward function, 

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
- introduce some weight-sharing between this embedding matrix and the $W_{out}$ matrix of the token generation layer

## 4. Improving Abstraction in Text Summarization(2018 EMNLP)
### problem
- word overlap metrics do not capture the abstractive nature
### contribution
- decouples the **extraction** and **generation** responsibilities of the decoder by factoring it into a **contextual network** and a **language model**.
- a mixed objective that jointly optimizes the **n-gram overlap** with the ground-truth summary while encouraging **abstraction**.

### model structure

#### ![improve_rl](D:\allkind\blog\martin6336.github.io\assets\post_image\improve_rl.png)contextual network the same as above

#### language model

The input of language model is the same as decoder at each time step $t$. Combing $h^{lm}_t$ with $[h_{t}^{d}|c_{t}^{e}| c_{t}^{d}]$, we have $h^{fuse}_t$
$$
\begin{aligned} f_{t} &=\operatorname{sigmoid}\left(W^{\operatorname{lm}}\left[r_{t} ; h_{3, t}^{\operatorname{lm}}\right]+b^{\operatorname{lm}}\right) \\ g_{t} &=W^{\mathrm{fuse}}\left(\left[r_{t} ; g_{t} \odot h_{3, t}^{\operatorname{lm}}\right]\right)+b^{\mathrm{fuse}} \\ h_{t}^{\mathrm{fuse}} &=\operatorname{ReLU}\left(g_{t}\right) \end{aligned}
$$

#### abstractive reward

changing the the reward function  $r(y)$.



## 5. SummaRuNNer(2017 AAAI)

### problem

### contribution

### model structure

![SummaRuNNer](D:\allkind\blog\martin6336.github.io\assets\post_image\SummaRuNNer.png)

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

We need to label sentences in the document for extractive training, which will bring loss. This paper propose a novel training technique to train SummaRuNNer abstractively. We **couple the initial model with a RNN decoder** that models the generation of abstractive summaries at training time only. We modify the first three equations of GRU based RNN, $s_{-1}$ is the summary representation computed at the last sentence of the sentence-level bidirectional RNN. It's the only bridge between the initial model and the coupled decoder.


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
where $N_s$ is the number of words in the reference summary.



