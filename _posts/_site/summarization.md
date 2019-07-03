# summarization

## 1,Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond(2016 coNLL)###

### problems to solve
- 和MT类似，但是seq2seq的两头没有一一对应关系
- 对信息的压缩 in a lossy manner，MT是loss-less

### contribution
- attentional encoder-decoder RNN
- propose novel models
- propose a new dataset

### model structure
1. Encoder-Decoder RNN with Attention. 基本模型依据[Bahdanau](https://arxiv.org/pdf/1409.0473v7.pdf "Neural machine translation by jointly learning to align and translate")提出的模型。
- encoder consists of a bidirectional GRU-RNN
- decoder consists of a uni-directional GRU-RNN($s_{i}=f(s_{i-1}, y_{i-1}, c_{i})$)
- attention mechanism over the source-hidden states
- a soft-max layer over target vocabulary to generate words
- 

$x^2+cos(\theta)$

$$\sum_{i=1}^n a_i=0$$