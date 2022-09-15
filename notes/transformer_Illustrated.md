# Transformer模型 图解
参考文档：
* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)


attention是一个有助于提高神经机器翻译应用程序性能的概念。在这篇文章中，我们将研究Transformer——一种利用注意力来提高这些模型训练速度的模型。Transformer在特定任务中优于Google神经机器翻译模型。然而，最大的好处来自Transformer如何适应并行化。事实上，Google Cloud建议使用Transformer作为使用其云TPU产品的参考模型。所以，让我们试着将模型分解，看看它是如何工作的。

Transformer在论文Attention is All You Need中提出。它的TensorFlow实现作为Tensor2Sensor软件包的一部分提供。哈佛大学的NLP小组创建了一个指南，用PyTorch实现对论文进行注释。在这篇文章中，我们将尝试将事情简单化一点，并逐一介绍这些概念，希望让没有深入了解主题的人更容易理解。

### A High-Level Look
让我们首先将模型视为单个黑盒。在机器翻译应用程序中，它将获取一种语言的句子，并输出另一种语言中的翻译。

![transformer ](https://jalammar.github.io/images/t/the_transformer_3.png)

Transformer的本质上是一个Encoder-Decoder的结构

![transformer structure](https://pic3.zhimg.com/80/v2-e0bbaaa73ab19da457d2c39242269cc2_1440w.jpg)

编码组件是一个编码器的堆栈（论文中堆叠了六个编码器，其中六个在彼此的顶部——数字六没有什么神奇之处，可以用其他排列进行实验）。解码组件是相同数量的解码器的堆栈。

![transformer structure2](https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)

编码器在结构上完全相同（但它们不共享权重）。每一层分为两个子层：
![transformer layer](https://jalammar.github.io/images/t/Transformer_encoder.png)


编码器的输入首先通过一个self-attention层，该层帮助编码器在编码特定单词时查看输入句子中的其他单词。我们将在后面的文章中进一步关注self-attention。
self-attention层的输出被馈送到前馈神经网络。完全相同的前馈网络独立地应用于每个位置。
解码器具有这两个层，但它们之间有一个attention layer，帮助解码器关注输入句子的相关部分（类似于seq2seq模型中的attention）。
![transformer decoder](https://jalammar.github.io/images/t/Transformer_decoder.png)


### 引入张量
既然我们已经看到了模型的主要组成部分，让我们开始看看各种向量/张量，以及它们如何在这些组成部分之间流动，从而将训练模型的输入转化为输出。
与一般的NLP应用程序一样，我们首先使用* [embedding算法](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca)将每个输入字转换为向量。

![transformer decoder](https://jalammar.github.io/images/t/embeddings.png)
*Each word is embedded into a vector of size 512. We'll represent those vectors with these simple boxes.*

embedding仅发生在最底部的编码器中。所有编码器共有的抽象是，它们接收每个大小为512的矢量列表–在底部编码器中，这将是字嵌入，但在其他编码器中，它将是直接位于下方的编码器的输出。这个列表的大小是我们可以设置的超参数——基本上是训练数据集中最长句子的长度。

在输入序列中嵌入单词后，每个单词都流经编码器的两层。
![transformer encoder_with_tensors](https://jalammar.github.io/images/t/encoder_with_tensors.png)


在这里，我们开始看到transformer的一个关键属性，即每个位置的字在编码器中通过其自己的路径流动。在self-attention层中，这些路径之间存在依赖关系。然而，前馈层不具有这些依赖性，因此，在流经前馈层时，可以并行执行各种路径。

### Encoding part
正如我们已经提到的，编码器接收向量列表作为输入。它通过将这些向量传递到“self-attention”层，然后传递到前馈神经网络，然后将输出向上发送到下一个编码器来处理该列表。
![transformer encoder_with_tensors_2](https://jalammar.github.io/images/t/encoder_with_tensors_2.png)
*The word at each position passes through a self-attention process. Then, they each pass through a feed-forward neural network -- the exact same network with each vector flowing through it separately.*



### Self-Attention at a High Level
不要被我胡扯“self-attention”这个词所愚弄，好像这是每个人都应该熟悉的概念。在阅读《Attention is All You Need》这篇文章之前，我个人从未遇到过这个概念。让我们总结一下它是如何工作的。

假设下面的句子是我们要翻译的输入句子：
  
*”The animal didn't cross the street because it was too tired”*

这个句子中的“it”指的是什么？它指的是street还是animal？这对人类来说是一个简单的问题，但对算法来说却不是那么简单。

当模型处理“it”这个词时，自我注意力允许它将“it”与“animal”联系起来。

当模型处理每个单词（输入序列中的每个位置）时，自我注意力允许它查看输入序列中其他位置，寻找有助于对该单词进行更好编码的线索。

如果您熟悉RNN，请考虑保持隐藏状态如何使RNN将其处理的先前单词/向量的表示与当前处理的单词/向量相结合。self-attention 是Transformer用来将其他相关词汇的“理解”转化为我们当前正在处理的词汇的方法。

![attention_visualization](https://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

*As we are encoding the word "it" in encoder #5 (the top encoder in the stack), part of the attention mechanism was focusing on "The Animal", and baked a part of its representation into the encoding of "it".*

### Self-Attention 详解
让我们先看看如何使用向量计算self-attention，然后再看看它是如何实际实现的——使用矩阵。

计算self-attention的**第一步**是从编码器的每个输入向量创建三个向量（在本例中，每个词的嵌入）。因此，对于每个单词，我们创建一个查询向量、一个关键字向量和一个值向量。这些向量是通过将嵌入乘以我们在训练过程中训练的三个矩阵来创建的。

请注意，这些新向量的维数小于嵌入向量。它们的维数为64，而嵌入和编码器输入/输出向量的维数为512。它们不必更小，这是一种使多头注意力（大部分）计算恒定的架构选择。

![transformer_self_attention_vectors](https://jalammar.github.io/images/t/transformer_self_attention_vectors.png)
*Multiplying x1 by the WQ weight matrix produces q1, the "query" vector associated with that word. We end up creating a "query", a "key", and a "value" projection of each word in the input sentence.*


“query”, “key”, and “value” 向量如何理解？
它们是用于计算和思考注意力的抽象概念。一旦你继续阅读下面的注意力计算方法，你将几乎了解所有这些向量所扮演的角色。


计算self-attention的**第二步**是计算score。假设我们正在计算本例中第一个单词“思考”的self-attention。我们需要对输入句子中的每个单词与这个单词进行评分。分数决定了当我们在某个位置编码一个单词时，在输入句子的其他部分上放置多少focus。

得分是通过将query vector与我们正在得分的各个词的key vector的点积dot product来计算的。因此，如果我们处理位置#1中单词的self-attention，第一个分数将是q1和k1的点积。第二个分数将为q1和k2的点乘积。
![transformer_self_attention_score](https://jalammar.github.io/images/t/transformer_self_attention_score.png)


**第三步和第四步**是将分数除以8（论文中使用的关键向量维度的平方根–64）。这会导致更稳定的梯度。这里可能有其他可能的值，但这是默认值），然后通过softmax操作传递结果。Softmax将分数标准化，使其全部为正，相加和为1。

![self-attention_softmax](https://jalammar.github.io/images/t/self-attention_softmax.png)

该softmax分数确定每个单词在该位置的表达量。显然，这个位置的单词将具有最高的softmax分数，但有时关注与当前单词相关的另一个单词是有用的。

**第五步**是将每个值向量乘以softmax得分（准备将它们相加）。这里的直觉是保持我们想要关注的单词的值不变，并淹没无关的单词（例如，通过将它们乘以0.001等微小数字）。

**第六步**是对加权值向量求和。这在该位置（对于第一个单词）产生自关注层的输出。

![self-attention-output](https://jalammar.github.io/images/t/self-attention-output.png)


self-attention计算到此结束。我们可以将得到的向量发送给前馈神经网络。然而，在实际实现中，这种计算是以矩阵形式进行的，以加快处理速度。现在让我们来看一下，我们已经看到了在单词层面上计算的直觉。

### self-Attention的矩阵计算

第一步是计算查询、键和值矩阵。我们通过将嵌入打包到矩阵X中，并将其乘以我们训练的权重矩阵（WQ、WK、WV）来实现这一点。
![self-attention-matrix-calculation](https://jalammar.github.io/images/t/self-attention-matrix-calculation.png)

*Every row in the X matrix corresponds to a word in the input sentence. We again see the difference in size of the embedding vector (512, or 4 boxes in the figure), and the q/k/v vectors (64, or 3 boxes in the figure)*

最后，由于我们处理的是矩阵，我们可以将第二步到第六步浓缩在一个公式中，以计算自我注意层的输出。

![self-attention-matrix-calculation-2](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)
*The self-attention calculation in matrix form*


### The Beast With Many Heads
本文通过添加一种称为“multi-headed”的attention机制，进一步完善了self-attention层。这从两个方面提高了attention层的性能：

它扩展了模型关注不同位置的能力。是的，在上面的例子中，z1包含一点点其他编码，但它可能由实际单词本身支配。如果我们翻译一句话，比如“动物没有过马路是因为它太累了”，那么知道“它”指的是哪个词会很有用。

它为关注层提供了多个“表示子空间”。正如我们接下来将看到的，对于多个注意力，我们不仅有一组，而且有多组查询/键/值权重矩阵（Transformer使用八个注意力头，因此每个编码器/解码器都有八组注意力）。这些集合中的每一个都被随机初始化。然后，在训练之后，每个集合用于将输入嵌入（或来自较低编码器/解码器的向量）投影到不同的表示子空间中。

![transformer_attention_heads_qkv](https://jalammar.github.io/images/t/transformer_attention_heads_qkv.png)
*With multi-headed attention, we maintain separate Q/K/V weight matrices for each head resulting in different Q/K/V matrices. As we did before, we multiply X by the WQ/WK/WV matrices to produce Q/K/V matrices.*


如果我们进行与上述相同的自我注意计算，只需使用不同的权重矩阵进行八次不同的计算，我们最终得到八个不同的Z矩阵

![transformer_attention_heads_z](https://jalammar.github.io/images/t/transformer_attention_heads_z.png)

这给我们留下了一点挑战。前馈层不需要八个矩阵——它只需要一个矩阵（每个词有一个向量）。所以我们需要一种方法把这八个压缩成一个矩阵。

我们如何做到这一点？我们浓缩矩阵，然后将它们乘以附加权重矩阵WO。

![transformer_attention_heads_weight_matrix_o](https://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png)

这几乎就是multi-headed self-attention的全部。我意识到这是相当多的矩阵。让我试着把它们放在一个视觉上，这样我们就可以在一个地方看到它们

![transformer_multi-headed_self-attention-recap](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)


既然我们已经谈到了attention heads，让我们重新回顾一下之前的例子，看看在我们的示例句子中编码“it”一词时，不同的attention heads集中在哪里：

![transformer_self-attention_visualization_2](https://jalammar.github.io/images/t/transformer_self-attention_visualization_2.png)
*As we encode the word "it", one attention head is focusing most on "the animal", while another is focusing on "tired" -- in a sense, the model's representation of the word "it" bakes in some of the representation of both "animal" and "tired".*

然而，如果我们把所有注意力集中在画面上，事情可能会变得更难解释：

![attention_visualization_3](https://jalammar.github.io/images/t/transformer_self-attention_visualization_3.png)



### 使用Positional Encoding表示序列的顺序
到目前为止，我们所描述的模型中缺少的一点是一种解释输入序列中单词顺序的方法。

为了解决这个问题，transformer向每个input embedding添加一个vector。这些vector遵循模型学习的特定模式，这有助于确定每个单词的位置，或序列中不同单词之间的距离。这里的直觉是，将这些值添加到embeddings中，一旦embedding vectors被投影到Q/K/V向量中，以及在点积注意期间，就可以在embedding vector之间提供有意义的距离。

![transformer_positional_encoding_vectors](https://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png)

*To give the model a sense of the order of the words, we add positional encoding vectors -- the values of which follow a specific pattern.*

如果我们假设嵌入的维数为4，则实际位置编码将如下所示：
![transformer_positional_encoding_example](https://jalammar.github.io/images/t/transformer_positional_encoding_example.png)
*A real example of positional encoding with a toy embedding size of 4*


模型看上去会是什么样子呢？
在下图中，每行对应于向量的位置编码。因此，第一行将是我们添加到输入序列中第一个单词的嵌入中的向量。每行包含512个值，每个值的值介于1和-1之间。我们对它们进行了颜色编码，以使图案可见。

![transformer_positional_encoding_large_example](https://jalammar.github.io/images/t/transformer_positional_encoding_large_example.png)
*对于嵌入大小为512（列）的20个字（行）的位置编码的真实示例。您可以看到，它看起来从中心一分为二。这是因为左半部分的值由一个函数（使用正弦）生成，右半部分由另一个函数生成（使用余弦）。然后将它们连接起来形成每个位置编码向量。*

本文描述了位置编码公式（第3.5节）。您可以在get_timing_signal_1d（）中看到生成位置编码的代码。这不是位置编码的唯一可能方法。然而，它的优点是能够缩放到看不见的序列长度（例如，如果要求我们的训练模型翻译比我们训练集中的任何句子都长的句子）。

2020年7月更新：上面显示的位置编码来自变压器的Transformer2Transformer实现。本文所示的方法稍有不同，它不直接连接，而是交织两个信号。下图显示了它的外观。下面是生成它的代码：
![attention-is-all-you-need-positional-encoding](https://jalammar.github.io/images/t/attention-is-all-you-need-positional-encoding.png)


### 残差网络
在继续之前，我们需要提到的编码器架构中的一个细节是，每个编码器中的每个子层（self-attention，ffnn）都有一个residual连接，然后是layer-normalization步骤。

![transformer_resideual_layer_norm](https://jalammar.github.io/images/t/transformer_resideual_layer_norm.png)


如果我们将vectors和与self-attention相关的layer-norm操作可视化，它将如下所示：

![transformer_resideual_layer_norm_2](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png)


这也适用于解码器的子层。如果我们考虑一个由2个堆叠编码器和解码器组成的Transformer，它看起来像这样：

![transformer_resideual_layer_norm_3](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)


### 解码器侧
既然我们已经介绍了编码器方面的大部分概念，我们基本上也知道解码器的组件是如何工作的。但让我们看看它们是如何协同工作的。

编码器从处理输入序列开始。然后，顶部编码器的输出被转换为一组关注向量K和V。这些向量将由每个解码器在其“encoder-decoder attention”层中使用，这有助于解码器关注输入序列中的适当位置：
![transformer_decoding_1](https://jalammar.github.io/images/t/transformer_decoding_1.gif)
*在完成编码阶段之后，我们开始解码阶段。解码阶段的每个步骤从输出序列中输出一个元素（在这种情况下为英语翻译句子）。*

以下步骤重复该过程，直到达到指示transformer decoder已完成其输出的特殊符号。每个步骤的输出在下一个时间步骤中被馈送到底部decoder，decoder像encoders一样将其解码结果冒泡出来。就像我们对encoders输入所做的那样，我们将位置编码嵌入并添加到这些解码器输入中，以指示每个单词的位置。

![transformer_decoding_2](https://jalammar.github.io/images/t/transformer_decoding_2.gif)


解码器中的self-attention层的工作方式与编码器中的略有不同：

在解码器中，仅允许self-attention关注输出序列中的较早位置。这是通过在self-attention计算的softmax步骤之前屏蔽未来位置（将其设置为-inf）来实现的。


“Encoder-Decoder Attention”层的工作原理与Encoder-Decoder Attention类似，只是它从其下一层创建查询矩阵，并从编码器堆栈的输出中获取键和值矩阵。

### The Final Linear and Softmax Layer
解码器堆栈输出浮点向量。我们如何把它变成一个词？这是最后一个线性层的工作，然后是Softmax层。

线性层是一个简单的完全连接的神经网络，它将解码器堆栈生成的向量投影成一个更大的向量，称为logits向量。

让我们假设我们的模型知道10000个独特的英语单词（我们的模型的“输出词汇表”），这些单词是从其训练数据集中学习的。这将使logits向量有10000个单元格宽——每个单元格对应于一个唯一单词的分数。这就是我们如何解释模型的输出，然后是线性层。

然后，softmax层将这些分数转化为概率（全部为正，合计为1.0）。选择具有最高概率的单元，并产生与其相关联的字作为该时间步的输出。

![transformer_decoder_output_softmax](https://jalammar.github.io/images/t/transformer_decoder_output_softmax.png)
*This figure starts from the bottom with the vector produced as the output of the decoder stack. It is then turned into an output word.*

### Training概述

既然我们已经通过一个经过训练的变换器涵盖了整个正向传递过程，那么看看训练模型的直觉将是有用的。

在训练过程中，未经训练的模型将经历完全相同的前向传递。但是，由于我们在标记的训练数据集上训练它，我们可以将其输出与实际的正确输出进行比较。

为了形象化，让我们假设我们的输出词汇表仅包含六个单词（“a”、“am”、“i”、“谢谢”、“学生”和“<eos>”（句子结尾的缩写）。
![vocabulary](https://jalammar.github.io/images/t/vocabulary.png)
*The output vocabulary of our model is created in the preprocessing phase before we even begin training.*

一旦定义了输出词汇表，我们就可以使用相同宽度的向量来表示词汇表中的每个单词。这也称为一个热编码。例如，我们可以使用以下向量表示单词“am”：
![vocabulary example](https://jalammar.github.io/images/t/one-hot-vocabulary-example.png)
*Example: one-hot encoding of our output vocabulary*

在本次回顾之后，让我们讨论模型的损失函数——我们在训练阶段优化的度量，以得到一个经过训练的、希望令人惊讶的精确模型。

### 损失函数
假设我们正在训练我们的模型。假设这是我们在培训阶段的第一步，我们正在一个简单的例子上进行培训——将“谢谢”翻译成“谢谢”。

这意味着，我们希望输出是一个概率分布，表示“谢谢”。但是，由于这个模型还没有被训练，这不太可能发生。

![vocabulary example](https://jalammar.github.io/images/t/transformer_logits_output_and_label.png)
*由于模型的参数（权重）都是随机初始化的，（未经训练的）模型为每个单元格/单词生成具有任意值的概率分布。我们可以将其与实际输出进行比较，然后使用反向传播调整所有模型的权重，使输出更接近期望的输出。*


你如何比较两种概率分布？我们只需从另一个中减去一个。有关详细信息，请参见cross-entropy and Kullback–Leibler散度。

但请注意，这是一个过于简化的示例。更现实地说，我们将使用一个超过一个单词的句子。例如–输入：“我是学生”，预期输出：“我是一名学生”。这实际上意味着，我们希望我们的模型连续输出概率分布，其中：

每个概率分布由一个宽度为vocab_size的向量表示（在我们的玩具示例中为6，但更实际的是30000或50000）

第一概率分布在与单词“i”相关联的单元处具有最高概率

第二概率分布在与单词“am”相关联的单元处具有最高概率

依此类推，直到第五个输出分布指示“＜语句结束＞”符号，该符号还具有10000个元素词汇表中与之相关的单元格。

![output_target_probability_distributions](https://jalammar.github.io/images/t/output_target_probability_distributions.png)
*The targeted probability distributions we'll train our model against in the training example for one sample sentence.*

在足够大的数据集上训练模型足够长的时间后，我们希望生成的概率分布如下所示：

![output_trained_model_probability_distributions](https://jalammar.github.io/images/t/output_trained_model_probability_distributions.png)
*希望经过培训，模型将输出我们期望的正确翻译。当然，这并不是这个短语是否是训练数据集的一部分的真实指示（参见：交叉验证）。请注意，每个位置都有一点概率，即使它不太可能是该时间步长的输出——这是softmax的一个非常有用的属性，有助于训练过程。*


现在，由于模型一次生成一个输出，我们可以假设模型从概率分布中选择概率最高的单词，并丢弃其余的。这是一种方法（称为贪婪解码）。另一种方法是保留前两个单词（例如，“I”和“a”），然后在下一步中，运行模型两次：一次假设第一个输出位置是单词“I”，另一次假设第二个输出位置为单词“a”，考虑到位置#1和352，无论哪个版本产生的错误都较小。我们对位置#2和353重复此操作…等等。这种方法称为“beam搜索”，在我们的示例中，beam_size为2（意味着在任何时候，两个部分假设（未完成的翻译）都保留在内存中），top_beam也为2（这意味着我们将返回两个翻译）。这两个超参数都可以进行实验。


### Go Forth And Transform
I hope you’ve found this a useful place to start to break the ice with the major concepts of the Transformer. If you want to go deeper, I’d suggest these next steps:

Read the Attention Is All You Need paper, the Transformer blog post (Transformer: A Novel Neural Network Architecture for Language Understanding), and the Tensor2Tensor announcement.
Watch Łukasz Kaiser’s talk walking through the model and its details
Play with the Jupyter Notebook provided as part of the Tensor2Tensor repo
Explore the Tensor2Tensor repo.
Follow-up works:

Depthwise Separable Convolutions for Neural Machine Translation
One Model To Learn Them All
Discrete Autoencoders for Sequence Models
Generating Wikipedia by Summarizing Long Sequences
Image Transformer
Training Tips for the Transformer Model
Self-Attention with Relative Position Representations
Fast Decoding in Sequence Models using Discrete Latent Variables
Adafactor: Adaptive Learning Rates with Sublinear Memory Cost




### 1、Transformer是什么？
Transformer的本质上是一个Encoder-Decoder的结构，如图所示：
<!-- ![transformer structure](https://pic3.zhimg.com/80/v2-e0bbaaa73ab19da457d2c39242269cc2_1440w.jpg) -->

在transformer没有诞生之前，大多数序列赚到模型（Encoder-Decoder）都是基于CNN和RNN的，在本篇文章前，已经介绍了一个Attention和Self-attention机制，而Transformer就是基于attention机制的，attention机制要比CNN和RNN好，好太多了，好在哪？Attention可以解决RNN及其变体存在的长距离依赖问题，也就是attention机制可以有更好的记忆力，能够记住更长距离的信息，另外最重要的就是attention支持并行化计算，这一点太关键了。而transformer模型就是完全基于attention机制的，他完全的抛弃了CNN和RNN的结构。



### 2、Transormer的模型结构
![transformer model](https://pic2.zhimg.com/80/v2-16f97513b2e0f0f25155bf140586c0ad_1440w.jpg)

Transformer就是一个基于多头注意力机制的模型，把注意力机制理解了，这个其实也就很简单了，之前已经详细介绍了注意力机制，在此就细说。

（之前在论文了写的关于Transformer Encoder模型）Transformer Encoder模型的输入是一句话的字嵌入表示和其对应的位置编码信息，模型的核心层是一个多头注意力机制。注意力机制最初应用在图像特征提取任务上，比如人在观察一幅图像时，并不会把图像中每一个部分都观察到，而是会把注意力放在重要的部分，后来研究人员把注意力机制应用到了NLP任务中，并取得了很好的效果。多头注意力机制就是使用多个注意力机制进行单独计算，以获取更多层面的语义信息，然后将各个注意力机制获取的结果进行拼接组合，得到最终的结果。Add&Norm层会把Multi-Head Attention层的输入和输出进行求和并归一化处理后，传递到Feed Forward层，最后会再进行一次Add&Norm处理，输出最终的词向量矩阵。


![transformer Encoder](https://pic3.zhimg.com/80/v2-39e89ab2ef733cce61508468657d1652_1440w.jpg) 

*Transformer Encoder模型结构*

### 3.Transormer的优缺点

优点：（1）虽然Transformer最终也没有逃脱传统学习的套路，Transformer也只是一个全连接（或者是一维卷积）加Attention的结合体。但是其设计已经足够有创新，因为其抛弃了在NLP中最根本的RNN或者CNN并且取得了非常不错的效果，算法的设计非常精彩，值得每个深度学习的相关人员仔细研究和品位。（2）Transformer的设计最大的带来性能提升的关键是将任意两个单词的距离是1，这对解决NLP中棘手的长期依赖问题是非常有效的。（3）Transformer不仅仅可以应用在NLP的机器翻译领域，甚至可以不局限于NLP领域，是非常有科研潜力的一个方向。（4）算法的并行性非常好，符合目前的硬件（主要指GPU）环境。

缺点：（1）粗暴的抛弃RNN和CNN虽然非常炫技，但是它也使模型丧失了捕捉局部特征的能力，RNN + CNN + Transformer的结合可能会带来更好的效果。（2）Transformer失去的位置信息其实在NLP中非常重要，而论文中在特征向量中加入Position Embedding也只是一个权宜之计，并没有改变Transformer结构上的固有缺陷。


![transformer Encoder](https://jalammar.github.io/images/t/transformer_decoding_2.gif) 




