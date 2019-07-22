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
计算context vector $c_t$时使用hierarchical attention，对上文的attention进一步处理而已

$$
P^{a}(j)=\frac{P_{w}^{a}(j) P_{s}^{a}(s(j))}{\sum_{k=1}^{N_{d}} P_{w}^{a}(k) P_{s}^{a}(s(k))}
$$